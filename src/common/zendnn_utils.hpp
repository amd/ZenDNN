/*******************************************************************************
* Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <sys/sysinfo.h>
#include <string>
#include "zendnn_logging.hpp"
#include "zendnn_helper.hpp"

#define DATA_FORMAT_NCHW 1
#define DATA_FORMAT_NHWC 0


#define ALIGNED_OFFSET          64
#ifndef ZENDNN_UTILS_HPP
#define ZENDNN_UTILS_HPP

#define BLIS_EXPERT     1

#ifdef BLIS_EXPERT
    #include <blis.h>
#endif

typedef unsigned int uint;

using namespace zendnn;

//Enable/disable Direct Convolution
//TODO: Make below two MACRO as ZENDNN_BLOCKED_FORMAT
#define ZENDNN_DIRECT_CONV              1
#define ZENDNN_BLOCKED_POOLING          1

#ifdef BLIS_EXPERT
//class to use blis expert interface
class blis_expert {

  public:
    rntm_t rntm;
    obj_t a, b, c;
    obj_t alpha, beta;
    num_t dt;

    blis_expert(int blis_num_threads, trans_t transa, trans_t transb,
                float bli_alpha = 1.0, float bli_beta=0.0,
                int supA = 0, int supB = 0) {

        dt = BLIS_FLOAT;
        bli_rntm_init(&rntm);
        bli_rntm_set_num_threads(blis_num_threads, &rntm);

        bli_obj_create(dt, 1, 1, 0, 0, &alpha);
        bli_obj_create(dt, 1, 1, 0, 0, &beta);
        bli_setsc(bli_alpha, 0.0, &alpha);
        bli_setsc(bli_beta, 0.0, &beta);

        bli_obj_set_conjtrans(transa, &a);
        bli_obj_set_conjtrans(transb, &b);

        bli_rntm_set_pack_a(supA, &rntm);
        bli_rntm_set_pack_b(supB, &rntm);
    }
};
#endif

zendnnEnv readEnv();

//This is for supporting future integration with streams
//TODO: Test with maximum no of possible streams and tune
//      ZEN_LIB_MEM_POOL_LIMIT accordingly
#define     ZEN_LIB_MEM_POOL_LIMIT          64

//ZEN_LIB_BUF_POOL_LIMIT define the limit for active buffers inside pool for
//  given Memory pool
//TODO: Test with increased limit and tune it accordingly
#define     ZEN_LIB_BUF_POOL_LIMIT       16


//zenLibBufPool structure holds zenLibBufPtr with its state 0, -1 and > 0
//  Possible states -1(not allocated),
//                  0(allocated and free)
//                  >0(occupied, no. of links with other node)
//  zenLibBufSize is size of pointed memory(no. of elements inside tensor)
typedef struct zenLibBufState {
    float               *zenLibBufPtr;
    int                 zenLibBufPtrStatus;
    unsigned long       zenLibBufSize;
} zenLibBufPool;


//class ZenLibMemoryPool holds description about memory pool with all buffer pointer
//  created inside pool.
class ZenLibMemoryPool {

    //zenLibMemPoolArr hold no. of memory pool exist, In case of multiple streams,
    //  each stream will have its own memory pool. Currently ZEN_LIB_MEM_POOL_LIMIT
    //  is the limit, can be made dynamic in future for supporting streams >
    //  ZEN_LIB_MEM_POOL_LIMIT. Single Memory pool object will be created for each
    //  stream, every call to getZenLibMemPool( <fixed index>) will return same
    //  object.
    //zenLibMemPoolCount hold the no of active memory pool
  private:
    static ZenLibMemoryPool *zenLibMemPoolArr[ZEN_LIB_MEM_POOL_LIMIT];
    static int zenLibMemPoolCount;
    //Initialize pool object with default values
    ZenLibMemoryPool() {
        zenLibBufPoolSize = 0;
        zenLibBufPoolLimit = ZEN_LIB_BUF_POOL_LIMIT;
        max_size = 1;
        max_size_enable = 0;
        //To enable/disable Reduced memory pool(default) OR Fixed memory
        //  pool buffer from env variable.
        //  Reduced memory pool buffer works with different size buffers in pool,
        //  Some models are exception to this. For those, in case of Reduced memory
        //  pool, some of the layers will use default memory allocation once we
        //  hit the pool limit with ZEN_LIB_BUF_POOL_LIMIT
        //  Otherwise works with Fixed memory pool buffers, this will works with
        //  finding max from increasing size as we go deeper into model.
        //  In future as a part of cost function max size will be calculated and
        //  used accordingly.
        max_size_enable = zendnn_getenv_int("ZENDNN_LIB_BUF_MAXSIZE_ENABLE", 0);
        //Getting max pool limit from env variable
        zenLibBufPoolLimit = zendnn_getenv_int("ZENDNN_LIB_BUF_POOL_LIMIT",
                                               ZEN_LIB_BUF_POOL_LIMIT);
        zenLibBufPoolLimit = (zenLibBufPoolLimit <=0)?1:zenLibBufPoolLimit;

        zenLibBufPoolArr = (zenLibBufPool *) malloc(zenLibBufPoolLimit * sizeof(
                               zenLibBufPool));

        for (int i=0; i<zenLibBufPoolLimit; i++) {
            zenLibBufPoolArr[i].zenLibBufPtr = NULL;
            zenLibBufPoolArr[i].zenLibBufPtrStatus = -1;
            zenLibBufPoolArr[i].zenLibBufSize = 0;
        }
    }

    //destroy Memory pool once done with usage
    ~ZenLibMemoryPool() {
        for (int i=0; i<zenLibBufPoolSize; i++) {
            free(zenLibBufPoolArr[i].zenLibBufPtr);
        }
        free(zenLibBufPoolArr);
    }

  public:
    //zenLibBufPoolArr will hold all ptr and state of buffers created
    //  in the pool
    //zenLibBufPtrStatus will hold the state of those buffers
    //  Possible states -1(not allocated),
    //                0(allocated and free)
    //                >0(buffer links with other node)
    zenLibBufPool     *zenLibBufPoolArr;

    //No. of allocated buffers inside pool
    unsigned int    zenLibBufPoolSize;

    //Max limit for active buffers inside pool
    unsigned int    zenLibBufPoolLimit;

    //max_size_enable will allocate all buffers in pool of size equal to size
    //  of the o/p first layer or running max with pool array
    //TODO: calulate max_size as part of cost function during graph analysis
    //  phase and then use it accordingly.
    int            max_size_enable;

    //max size of allocated buffers in the pool
    unsigned long     max_size;

    //Get Memory pool pointer from Global array of memory pool based on index
    //Create ZenMemPool object, if not created corresponding to that index
    static ZenLibMemoryPool *getZenLibMemPool(int index) {

        bool flag = false;
        #pragma omp critical
        {
            //ZEN_LIB_MEM_POOL_LIMIT is the hard limit on the total no. of ZenLibMemoryPool
            //TODO: Need to tune ZEN_LIB_MEM_POOL_LIMIT based on the available memory or
            //make it grow dynamically
            if (index >= ZEN_LIB_MEM_POOL_LIMIT) {
                flag = true;
            }
            else if (!zenLibMemPoolArr[index]) {
                zenLibMemPoolArr[index] = new ZenLibMemoryPool();
                zenLibMemPoolCount++;
            }
        }
        if (flag) {
            return NULL;
        }
        else {
            return zenLibMemPoolArr[index];
        }
    }

    //Free zenLibMemPoolArr based on index passed
    static void freeZenLibMemPool(int index) {

        #pragma omp critical
        {
            if (index < ZEN_LIB_MEM_POOL_LIMIT && zenLibMemPoolArr[index]) {
                delete zenLibMemPoolArr[index];
                zenLibMemPoolCount--;
            }
        }
    }

    //Reset status of all buffers as free at
    //the start of graph execution.
    void resetLibPoolStatus() {
        for (int i=0; i<zenLibBufPoolSize; i++) {
            zenLibBufPoolArr[i].zenLibBufPtrStatus = 0;
        }
    }
    //Acquire buffer from the given pool object. If pool is not
    //  initialized or buffer is not free, create buffer and
    //  add to the pool.
    int acquireZenLibPoolBuf(float **output, unsigned long out_size, int outlinks) {

        int return_flag = 0;
        #pragma omp critical
        {
            /*
                    if (reset && zenLibBufPoolSize) {
                        resetLibPoolStatus();
                    }
            */
            int acquire_flag = 0;
            int free_flag = 0;

            // Search for free buffer in pool for buffer_size >= out_size
            for (int i=0; i<zenLibBufPoolSize; i++) {
                if (zenLibBufPoolArr[i].zenLibBufPtrStatus == 0) {

                    free_flag = 1;

                    //Go to next free buffer when out_size is more
                    //  than buffer_size of pool at given offset.
                    unsigned long buffer_size = zenLibBufPoolArr[i].zenLibBufSize;
                    if (out_size > buffer_size) {
                        zenLibBufPoolArr[i].zenLibBufPtr = (float *) realloc(
                                                               zenLibBufPoolArr[i].zenLibBufPtr, out_size);
                        if (zenLibBufPoolArr[i].zenLibBufPtr == NULL) {
                            continue;
                        }
                        zenLibBufPoolArr[i].zenLibBufSize = out_size;
                    }
                    *output = zenLibBufPoolArr[i].zenLibBufPtr;
                    zenLibBufPoolArr[i].zenLibBufPtrStatus = outlinks;
                    acquire_flag = 1;
                    zendnnInfo(ZENDNN_ALGOLOG,
                               "\nLIB-MEM-POOL: Acquired libBufPool Ptr[", i,
                               "] pointed to size(no. of elements)",
                               buffer_size, "\n");

                    break;
                }
            }
            //If requested buffer not found in pool, go ahead and create
            //  new buffer inside pool.
            if (!acquire_flag) {

                if (zenLibBufPoolSize == zenLibBufPoolLimit) {
                    if (free_flag) {
                        zendnnInfo(ZENDNN_ALGOLOG,
                                   "\nLIB-MEM-POOL: Requested buffer from ZenLibMemPool, But Falling back to default allocation as out_size > available buffer_size inside Pool\n");
                    }
                    else {
                        zendnnInfo(ZENDNN_ALGOLOG,
                                   "\nLIB-MEM-POOL: Requested buffer from ZenLibMemPool, But Falling back to default allocation as zenLibBufPoolSize == ZEN_LIB_BUF_POOL_LIMIT\n");
                    }
                    return_flag = 1;
                }
                else {
                    unsigned int poolOffset = zenLibBufPoolSize;
                    unsigned long size;

                    //Set max_size based on current layer output dimension
                    //  and ZEN_LIB_BUF_SIZE_FACTOR, Most of the cases Output
                    //  dimension goes down after first layer. However few
                    //  models are exception to this.
                    //max_size required can be computed during first run
                    //  for graph execution and same can be used for Buffer
                    //  allocation. But this will not give optimal performance
                    //  for first graph execution.
                    //TODO: Compute max_size as part of cost function during
                    //  graph analysis phase
                    if (out_size > max_size) {
                        max_size = out_size;
                    }

                    //max_size_enable will create all the buffers with increasing
                    //  size inside the pool
                    if (max_size_enable) {
                        size = max_size;
                    }
                    else {
                        size = out_size;
                    }
                    zenLibBufPoolArr[poolOffset].zenLibBufPtr = (float *) aligned_alloc(
                                ALIGNED_OFFSET, sizeof(
                                    float) * size);

                    if (zenLibBufPoolArr[poolOffset].zenLibBufPtr == NULL) {
                        return_flag = 1;
                    }
                    else {
                        zenLibBufPoolArr[poolOffset].zenLibBufSize = size;
                        *output = zenLibBufPoolArr[poolOffset].zenLibBufPtr;
                        zenLibBufPoolArr[poolOffset].zenLibBufPtrStatus = outlinks;
                        acquire_flag = 1;
                        zenLibBufPoolSize++;
                        zendnnInfo(ZENDNN_ALGOLOG,
                                   "\nLIB-MEM-POOL: Allocation done for Buffer in Pool of size = ",
                                   size, " elements", " zenLibBufPoolCount = ",
                                   zenLibBufPoolSize-1, "\n",
                                   "LIB-MEM-POOL: Acquired LibBufPool Ptr[", poolOffset,
                                   "] pointed to size(no. of elements)",
                                   size, "\n");
                    }
                }
            }
        }
        if (return_flag) {
            return 1;
        }
        else {
            return 0;
        }


    }
    //This will update the state of Memory pool by decrementing
    //zenLibBufPtrStatus based on the input buffer comparison.
    void zenLibMemPoolFree(float *buffer) {
        #pragma omp critical
        {
            for (int i=0; i<zenLibBufPoolSize; i++) {
                float *buffer_array = zenLibBufPoolArr[i].zenLibBufPtr;
                if (buffer == buffer_array) {
                    zenLibBufPoolArr[i].zenLibBufPtrStatus--;
                    break;
                }

            }
        }

    }
};
#endif
