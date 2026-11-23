# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""The A/B gate (ab/ab.py) on corpus outputs: runs that differ only in log times and pointers are
identical, and a changed byte, plan line, log line (its module and level too), layout, verdict or a
missing case is named; a run that compares nothing fails; size counts the archive's objects only (by
their bytes), names the kernel functions of one build only and fails on a tree that lacks the
archive's objects; pack times the same cases in every round or fails.  With ZENDNNL_AB_CORPUS naming
an ab_corpus binary, also what its records hold."""

import contextlib
import io
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from native_brgemm_tuning.ab import ab as A  # noqa: E402

CALLS = """\
BEGIN bf16/M1
[API    ][verbose][0.001421]:[MATMUL.NATIVE_BRGEMM PACK] weight=0x5654a8 wei=bf16 K=4 N=32 into=in_place
[API    ][verbose][0.001803]:[MATMUL.NATIVE_BRGEMM] M=1 N=32 K=4 pack_nr=32 nr=32 gemv=bf16 team=1
[API    ][info   ][0.001820]:LOWOHA matmul_direct: M=1, N=32, K=4, kernel=native_brgemm, weight_cache=2
[API    ][verbose][0.001903]:[MATMUL.NATIVE_BRGEMM] M=1 N=32 K=4 pack_nr=32 nr=32 gemv=bf16 team=1
CASE bf16/M1 st=1 out=00000000000000aa again=00000000000000aa wei=00000000000000bb
BEGIN s8c/M8
[COMMON ][error  ][0.000131]:matmul_direct: native BRGEMM did not serve the call (a reason)
CASE s8c/M8 st=-1 out=00000000000000cc again=00000000000000cc wei=00000000000000dd
SUMMARY calls cases=2 served=1
"""

LAYOUT = """\
LAYOUT bf16/K1/N32/tB0/pad0/t1 st=1 bytes=128 hash=0000000000000001
UNPACK bf16/K1/N32/tB0/pad0/t1 hash=0000000000000002
LAYOUT s4/K8/N5/tB1/pad3/t1 st=1 bytes=128 hash=0000000000000003
SUMMARY layout cases=2
"""

RULES = """\
RULE wei=bf16/src=bf16/dst=f32/q=none/K1/sg0 taken=1 st=1 why=
[API    ][info   ][0.0004]:Reorder quantization: using dynamic symmetric quantization
RULE wei=s8/src=f16/dst=f32/q=none/K1/sg0 taken=1 st=-1 why=s8 weight with a source other than u8 / s8
SUMMARY rules cases=2
"""


class TestParse(unittest.TestCase):
    def test_calls_take_plan_and_log_lines_without_time_or_pointers(self):
        cases = A.parse_calls(CALLS)
        self.assertEqual(list(cases), ["bf16/M1", "s8c/M8"])
        c = cases["bf16/M1"]
        self.assertEqual(c["fields"], {"st": "1", "out": "00000000000000aa", "again": "00000000000000aa",
                                       "wei": "00000000000000bb"})
        plan = "[API    ][verbose][MATMUL.NATIVE_BRGEMM] M=1 N=32 K=4 pack_nr=32 nr=32 gemv=bf16 team=1"
        self.assertEqual(c["plan"], [plan] * 2)
        pack = "[API    ][verbose][MATMUL.NATIVE_BRGEMM PACK] weight=0x? wei=bf16 K=4 N=32 into=in_place"
        lowoha = "[API    ][info   ]LOWOHA matmul_direct: M=1, N=32, K=4, kernel=native_brgemm, weight_cache=2"
        self.assertEqual(c["log"], [pack, lowoha])
        self.assertEqual(cases["s8c/M8"]["plan"], [])
        self.assertEqual(A.summarize_calls(cases), "2 cases, 1 served, 2 plan lines")

    def test_records(self):
        layout = A.parse_records(LAYOUT, A.RECORD_KINDS["layout"])
        self.assertEqual(layout[("UNPACK", "bf16/K1/N32/tB0/pad0/t1")], "hash=0000000000000002")
        self.assertEqual(A.summarize_records(layout), "2 layout, 1 unpack")
        rules = A.parse_records(RULES, A.RECORD_KINDS["rules"])
        self.assertEqual(len(rules), 2)
        self.assertEqual(A.summarize_records(rules), "2 rule (1 served)")

    def test_size_totals(self):
        out = ("   text\t   data\t    bss\t    dec\t    hex\tfilename\n"
               "  12345\t      8\t      0\t  12353\t   3041\ta.o (ex lib.a)\n"
               "  12345\t      8\t      0\t  12353\t   3041\t(TOTALS)\n")
        self.assertEqual(A.parse_size_totals(out), (12345, 8, 0))
        with self.assertRaises(RuntimeError):
            A.parse_size_totals("text data bss\n")


class TestCompare(unittest.TestCase):
    def test_times_and_pointers_do_not_count(self):
        other = CALLS.replace("0.0018", "0.0091").replace("0x5654a8", "0x7f00ee")
        self.assertEqual(A.compare_mode("calls", CALLS, other)[1], [])
        self.assertEqual(A.compare_mode("layout", LAYOUT, LAYOUT)[1], [])

    def test_a_changed_output_byte_plan_or_log_line_is_named(self):
        diffs = A.compare_mode("calls", CALLS, CALLS.replace("again=00000000000000aa", "again=00000000000000ab"))[1]
        self.assertEqual(diffs, ["bf16/M1: again 00000000000000aa -> 00000000000000ab"])
        diffs = A.compare_mode("calls", CALLS, CALLS.replace("nr=32 gemv", "nr=16 gemv", 1))[1]
        self.assertEqual(diffs, ["bf16/M1: plan line 1: nr=32 -> nr=16"])
        diffs = A.compare_mode("calls", CALLS, CALLS.replace("into=in_place", "into=cache"))[1]
        self.assertEqual(diffs, ["bf16/M1: log line 1: into=in_place -> into=cache"])
        diffs = A.compare_mode("calls", CALLS, CALLS.replace("(a reason)", "(other reasons)"))[1]
        self.assertEqual(diffs, ["s8c/M8: log line 1: (a -> (other, reason) -> reasons)"])
        diffs = A.compare_mode("calls", CALLS, CALLS.replace("(a reason)", "(the reason why)"))[1]
        served = "[COMMON ][error  ]matmul_direct: native BRGEMM did not serve the call"
        self.assertEqual(diffs, [f"s8c/M8: log line 1: {served} (a reason) -> {served} (the reason why)"])
        extra = CALLS.replace("team=1\nCASE", "team=1\n[API    ][verbose][1]:x\nCASE")
        self.assertEqual(A.compare_mode("calls", CALLS, extra)[1], ["bf16/M1: log lines 2 -> 3"])

    def test_a_changed_level_or_module_is_named(self):
        diffs = A.compare_mode("calls", CALLS, CALLS.replace("[API    ][info   ]", "[API    ][error  ]"))[1]
        self.assertEqual(len(diffs), 1)
        self.assertTrue(diffs[0].startswith("bf16/M1: log line 2: [API    ][info   ]LOWOHA"), diffs)
        self.assertIn("-> [API    ][error  ]LOWOHA", diffs[0])
        diffs = A.compare_mode("calls", CALLS, CALLS.replace("[COMMON ][error  ]", "[API    ][error  ]"))[1]
        self.assertEqual(len(diffs), 1)
        self.assertTrue(diffs[0].startswith("s8c/M8: log line 1: [COMMON ][error  ]matmul_direct"), diffs)
        plan = CALLS.replace("[API    ][verbose][0.001803]", "[API    ][info   ][0.001803]")
        diffs = A.compare_mode("calls", CALLS, plan)[1]
        self.assertEqual(len(diffs), 1)
        self.assertTrue(diffs[0].startswith("bf16/M1: plan line 1: "), diffs)

    def test_a_missing_case_layout_or_verdict_is_named(self):
        cut = CALLS[:CALLS.index("BEGIN s8c")]
        self.assertEqual(A.compare_mode("calls", CALLS, cut)[1], ["s8c/M8: only in A"])
        diffs = A.compare_mode("layout", LAYOUT, LAYOUT.replace("hash=0000000000000003", "hash=0000000000000004"))[1]
        changed = "st=1 bytes=128 hash=0000000000000003 -> st=1 bytes=128 hash=0000000000000004"
        self.assertEqual(diffs, [f"LAYOUT s4/K8/N5/tB1/pad3/t1: {changed}"])
        diffs = A.compare_mode("rules", RULES, RULES.replace("st=-1 why=s8", "st=1 why=s8"))[1]
        self.assertEqual(len(diffs), 1)
        self.assertTrue(diffs[0].startswith("RULE wei=s8/src=f16/dst=f32/q=none/K1/sg0: taken=1 st=-1"))


class TestRun(unittest.TestCase):
    """ab.py run on two stand-in binaries that print fixed outputs."""

    def binary(self, folder, name, texts):
        for mode, text in texts.items():
            with open(os.path.join(folder, f"{name}.{mode}"), "w") as f:
                f.write(text)
        path = os.path.join(folder, name)
        with open(path, "w") as f:
            f.write(f'#!/bin/sh\n[ "$ZENDNNL_MATMUL_ALGO" = 10 ] || exit 3\ncat "{path}.$1"\n')
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def run_ab(self, a, b, out, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            status = A.main(["run", a, b, "--out", out, *extra])
        return status, buf.getvalue()

    def test_identical_and_differing_builds(self):
        with tempfile.TemporaryDirectory() as d:
            texts = {"calls": CALLS, "layout": LAYOUT, "rules": RULES}
            a = self.binary(d, "a", texts)
            b = self.binary(d, "b", texts)
            status, out = self.run_ab(a, b, os.path.join(d, "out"))
            self.assertEqual(status, 0, out)
            self.assertIn("calls: 2 cases, 1 served, 2 plan lines: identical", out)
            self.assertTrue(os.path.isfile(os.path.join(d, "out", "b.rules.txt")))
            c = self.binary(d, "c", dict(texts, layout=LAYOUT.replace("bytes=128 hash=0000000000000001",
                                                                       "bytes=128 hash=00000000000000ff")))
            status, out = self.run_ab(a, c, os.path.join(d, "out"))
            self.assertEqual(status, 1, out)
            self.assertIn("layout: 2 layout, 1 unpack: 1 differ", out)

    def test_a_run_that_compares_nothing_fails(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.binary(d, "a", {"calls": CALLS, "layout": LAYOUT, "rules": RULES})
            out_dir = os.path.join(d, "out")
            for modes in ("", ","):
                status, out = self.run_ab(a, a, out_dir, "--modes", modes)
                self.assertEqual(status, 2, out)
                self.assertIn("no mode", out)
            # A filter that matches no case: the runs hold their summaries only.
            empty = self.binary(d, "e", {"calls": "SUMMARY calls cases=0 served=0\n",
                                         "layout": "SUMMARY layout cases=0\n",
                                         "rules": "SUMMARY rules cases=0\n"})
            status, out = self.run_ab(empty, empty, out_dir, "--filter", "no such case")
            self.assertEqual(status, 2, out)
            for mode in A.MODES:
                self.assertIn(f"{mode}: the A run holds no case for this filter: nothing compared", out)
            status, out = self.run_ab(a, a, out_dir, "--modes", "rules")
            self.assertEqual(status, 0, out)


@unittest.skipUnless(all(shutil.which(t) for t in ("cc", "ar", "nm", "size")), "needs cc, ar, nm and size")
class TestSize(unittest.TestCase):
    """ab.py size on scratch build trees: the archive's objects only, and the kernel functions by name."""

    KERNEL = "lowoha_operators/matmul/matmul_native/brgemm/kernel/k.cpp.o"
    PACK = "lowoha_operators/matmul/matmul_native/common/p.cpp.o"
    OTHER = "common/other.cpp.o"

    def compile(self, path, funcs):
        """An object of funcs at path, compiled in its directory: the bytes of two objects of the same
        name differ by their functions only."""
        folder, name = os.path.split(path)
        os.makedirs(folder, exist_ok=True)
        with open(path + ".c", "w") as f:
            f.write("".join(f"int {fn}(int x) {{ return x * {i + 3} + 1; }}\n" for i, fn in enumerate(funcs)))
        subprocess.run(["cc", "-c", "-O1", "-o", name, name + ".c"], cwd=folder, check=True)
        os.remove(path + ".c")

    def tree(self, root, kernels, twin=False):
        """A build tree whose archive holds a kernel object of the functions kernels, a pack object and an
        object outside matmul_native; beside them the shared library's copy of the kernel object, the
        object of a removed kernel source, and the stale object of a kernel source of the same name
        that moved.  With twin, the archive also holds a second kernel object, and a directory walked
        before the kernel directory the stale object of another kernel source of the same name, as
        large as the kernel object and of other bytes."""
        src = os.path.join(root, "zendnnl", "src")
        target = os.path.join(src, "CMakeFiles", "zendnnl_archive.dir")
        live = [os.path.join(target, rel) for rel in (self.KERNEL, self.PACK, self.OTHER)]
        for path, funcs in zip(live, (kernels, ["pack"], ["other"])):
            self.compile(path, funcs)
        self.compile(os.path.join(src, "CMakeFiles", "zendnnl.dir", self.KERNEL), kernels + ["shared_copy"])
        kernel_dir = os.path.dirname(os.path.join(target, self.KERNEL))
        self.compile(os.path.join(kernel_dir, "removed.cpp.o"), ["kernel_removed"])
        self.compile(os.path.join(os.path.dirname(kernel_dir), "k.cpp.o"), ["moved_1", "moved_2", "moved_3"])
        if twin:
            live.append(os.path.join(kernel_dir, "k2.cpp.o"))
            self.compile(live[-1], ["kernel_base"])
            stale = os.path.join(os.path.dirname(kernel_dir), "a", "k.cpp.o")
            self.compile(stale, [fn.replace("kernel", "stale_") for fn in kernels])
            self.assertEqual(os.path.getsize(stale), os.path.getsize(live[0]))
        subprocess.run(["ar", "qc", os.path.join(src, "libzendnnl_archive.a"), *live], check=True)
        return live

    def test_the_archive_objects_and_the_kernel_functions(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a"), os.path.join(d, "b")
            live = self.tree(a, ["kernel_one", "kernel_two"])
            self.tree(b, ["kernel_one", "kernel_three"])
            self.assertEqual(A.objects_under(a, A.NATIVE_DIR), sorted(live[:2]))
            self.assertEqual(A.objects_under(a, A.KERNEL_DIR), [live[0]])
            report = A.size_report(a)
            self.assertEqual(report["kernel symbols"], {"kernel_one", "kernel_two"})
            self.assertEqual(report["matmul_native text"], A.size_totals(live[:2])[0])
            self.assertEqual(report["library text"], A.size_totals(live)[0])
            for extra, status in (([], 0), (["--same-kernels"], 1)):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    self.assertEqual(A.main(["size", a, b, *extra]), status, buf.getvalue())
                out = buf.getvalue()
                self.assertIn("kernel symbols                  2            2         +0", out)
                self.assertIn("  only in A: kernel_two\n  only in B: kernel_three\n", out)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(A.main(["size", a, a, "--same-kernels", "--budget", "0"]), 0)
            self.assertNotIn("only in", buf.getvalue())

    def size(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            status = A.main(["size", *args])
        return status, buf.getvalue()

    def test_an_object_is_told_from_a_stale_one_of_its_name_and_size_by_its_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a"), os.path.join(d, "b")
            live = self.tree(a, ["kernel_one", "kernel_two"], twin=True)
            self.tree(b, ["kernel_one", "kernel_six"], twin=True)
            self.assertEqual(A.objects_under(a, A.KERNEL_DIR), sorted([live[0], live[3]]))
            self.assertEqual(A.size_report(a)["kernel symbols"], {"kernel_one", "kernel_two", "kernel_base"})
            status, out = self.size(a, b, "--same-kernels")
            self.assertEqual(status, 1, out)
            self.assertIn("  only in A: kernel_two\n  only in B: kernel_six\n", out)
            # A stale copy of the same bytes outside the kernel directory leaves the place of the object open.
            stale = os.path.join(os.path.dirname(os.path.dirname(live[0])), "a", "k.cpp.o")
            shutil.copyfile(live[0], stale)
            with self.assertRaisesRegex(RuntimeError, "remove the stale ones"):
                A.objects_under(a, A.KERNEL_DIR)
            self.assertEqual(self.size(a, b)[0], 2)

    def test_a_tree_without_the_objects_of_its_archive_fails(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a"), os.path.join(d, "b")
            self.tree(a, ["kernel_one"])
            self.tree(b, ["kernel_two"])
            for root in (a, b):
                shutil.rmtree(os.path.join(root, "zendnnl", "src", "CMakeFiles"))
            status, out = self.size(a, b, "--same-kernels")
            self.assertEqual(status, 2, out)
            self.assertIn("which no object under", out)
            self.assertNotIn("kernel symbols", out)


class TestPack(unittest.TestCase):
    """ab.py pack on stand-in binaries: the times of the same cases in every round, else status 2."""

    TIMES = ("PACK bf16/K4096/N4096/tB1/pad0/t1 median_ms=2.0 min_ms=1.9\n"
             "UNPACK bf16/K4096/N4096/tB1/pad0/t1 median_ms=4.0 min_ms=3.9\n"
             "PACK s4/K4096/N4096/tB1/pad0/t1 median_ms=1.0 min_ms=0.9\n")
    binary = TestRun.binary

    def pack(self, a, b, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            status = A.main(["pack", a, b, "--rounds", "2", *extra])
        return status, buf.getvalue()

    def test_the_same_cases_are_timed(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.binary(d, "a", {"pack": self.TIMES})
            b = self.binary(d, "b", {"pack": self.TIMES.replace("median_ms=2.0", "median_ms=3.0")
                                     .replace("median_ms=4.0", "median_ms=1.0")})
            status, out = self.pack(a, b)
            self.assertEqual(status, 0, out)
            self.assertIn("pack: geomean B/A 1.225 over 2 cases", out)
            self.assertIn("unpack: geomean B/A 0.250 over 1 cases", out)
            self.assertIn("unpack bf16/K4096/N4096/tB1/pad0/t1", out)
            # A run without the unpack's time times other cases.
            packs = self.binary(d, "p", {"pack": self.TIMES.replace("UNPACK", "#")})
            status, out = self.pack(a, packs)
            self.assertEqual(status, 2, out)
            self.assertIn("unpack bf16/K4096/N4096/tB1/pad0/t1 in one of them only", out)

    def test_a_run_without_its_cases_fails(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.binary(d, "a", {"pack": self.TIMES})
            empty = self.binary(d, "e", {"pack": ""})
            status, out = self.pack(empty, empty, "--filter", "no such case")
            self.assertEqual(status, 2, out)
            self.assertIn("pack: the A run holds no case for this filter: nothing timed", out)
            self.assertEqual(self.pack(a, a, "--rounds", "0")[0], 2)
            lacks = self.binary(d, "l", {"pack": self.TIMES.splitlines(keepends=True)[0]})
            adds = self.binary(d, "x", {"pack": self.TIMES + "PACK f32/K1/N1/tB1/pad0/t1 median_ms=1 min_ms=1\n"})
            for b, odd in ((lacks, "s4/K4096/N4096/tB1/pad0/t1"), (adds, "f32/K1/N1/tB1/pad0/t1")):
                status, out = self.pack(a, b)
                self.assertEqual(status, 2, out)
                self.assertIn("pack: round 1 of B times other cases than round 1 of A", out)
                self.assertIn(f"{odd} in one of them only", out)
                self.assertNotIn("geomean", out)


@unittest.skipUnless(os.environ.get("ZENDNNL_AB_CORPUS"), "ZENDNNL_AB_CORPUS names no ab_corpus binary")
class TestCorpusRun(unittest.TestCase):
    """The ab_corpus binary that ZENDNNL_AB_CORPUS names, on a few cases."""

    def records(self, mode, text):
        out = A.run_corpus(os.environ["ZENDNNL_AB_CORPUS"], mode, 4, ["--filter", text])
        return A.parse_calls(out) if mode == "calls" else A.parse_records(out, A.RECORD_KINDS[mode])

    def test_a_filter_selects_verdicts_by_their_names(self):
        name = "wei=s8/src=u8/dst=f32/q=none/K64/sg1"
        for text, names in ((name, [name, name + "/algo10"]), (name + "/algo10", [name + "/algo10"])):
            self.assertEqual([n for _, n in self.records("rules", text)], names)

    def test_weight_scale_groups_that_do_not_split_k_are_refused(self):
        verdicts = self.records("rules", "/q=none/K")
        for wei, why in (("s8/src=u8", "per-group weight scales over groups of other than whole K quads"),
                         ("s4/src=s8", "an s4 weight in groups of other than whole K quads")):
            for K in (1, 6, 64, 96):
                fields = verdicts[("RULE", f"wei={wei}/dst=f32/q=none/K{K}/sg-1")]
                self.assertEqual(fields, f"taken=1 st=-1 why={why}", (wei, K))

    def test_a_case_holds_both_statuses_and_the_source_quantization(self):
        cases = self.records("calls", "s8c/M8/N96/K262/tA0/tB1/pad0/src=bf16:u8/dst=bf16/")
        self.assertEqual(len(cases), 1)
        fields = next(iter(cases.values()))["fields"]
        self.assertEqual((fields["st"], fields["st2"]), ("1", "1"))
        self.assertEqual(fields["sq"], fields["sq2"])
        cases = self.records("calls", "bf16/M8/N96/K263/tA0/tB1/pad0/src=bf16/dst=bf16/")
        self.assertTrue(cases)
        for case in cases.values():
            self.assertEqual(case["fields"]["st2"], case["fields"]["st"])
            self.assertNotIn("sq", case["fields"])


if __name__ == "__main__":
    unittest.main()
