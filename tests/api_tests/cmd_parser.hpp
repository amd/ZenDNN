/*******************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#ifndef _ZENDNN_TEST_CMD_PARSER_
#define _ZENDNN_TEST_CMD_PARSER_

#include <exception>
#include <string>
#include <cstdint>
#include <iostream>

#include "zendnn.hpp"

namespace test_utils {
    // command line args, initialized with their default values
    struct CommandLineArgs {
        std::string      engine_kind_str               = "cpu";
        std::string      emb_precision_str             = "fp32";
    };

    class BadCommandLineArg : public std::exception {
    public:
        BadCommandLineArg(std::string err_msg){
            what_msg = err_msg;
        }
        const char* what() const noexcept override {
            return what_msg.c_str();
        }
    private:
        std::string  what_msg;
    };

    class CommandLineParser {
    public:
        void                       ParseArgs(int argc, char* argv[]);
        void                       SanityCheck();
        CommandLineArgs            GetArgs();
    private:
        CommandLineArgs cmd_args;
    };

    void CommandLineParser::ParseArgs(int argc, char* argv[]){
        int           count    = 0;
        std::string   key;
        while (count < argc){
            if (argv[count][0] == '-'){
                // get the key
                if (argv[count][1] == '-')
                    key = std::string(argv[count]).substr(2);
                else
                    key = std::string(argv[count]).substr(1);

                count++;

                if (key == "engine")
                    cmd_args.engine_kind_str = std::string(std::string(argv[count++]));
                else if (key == "emb-precision")
                    cmd_args.emb_precision_str = std::string(std::string(argv[count++]));
                else {
                    std::string err_str;
                    err_str  = std::string("CommandLineArgs:invalid argument <");
                    err_str +=  key;
                    err_str += ">";

                    throw BadCommandLineArg(err_str);
                }
            }
            else {
                count++;
            }
        }
    }

    void CommandLineParser::SanityCheck() {
        std::string invalid_args = std::string();

        bool invalid_engine_kind = (cmd_args.engine_kind_str != "cpu");
        if (invalid_engine_kind){
            invalid_args += " --engine";
        }

        bool invalid_emb_precision = ((cmd_args.emb_precision_str != "fp32") &&
                                      (cmd_args.emb_precision_str != "bf16"));

        if (invalid_emb_precision){
            invalid_args += " --emb_precision";
        }

        if (invalid_args.length()){
            std::string err_msg;
            err_msg = "invalid values for " + invalid_args;
            throw BadCommandLineArg(err_msg);
        }
    }

    // get args
    CommandLineArgs CommandLineParser::GetArgs() {
        return cmd_args;
    }

} // namespace cmd


#endif
