# checks

Run-time cost is allowed only in the checked and debug builds. A plain release build carries none. Declared facts are checked by build level (extern: declared facts §1).

Checks come after the runtime runs: first it runs, then its checks.
