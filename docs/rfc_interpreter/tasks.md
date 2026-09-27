# tasks

## 1. Two facts

A call's task is two independent facts, joined componentwise: whether it suspends, and whether it is heavy.

## 2. Heavy is taint

Heaviness spreads along every dependency: whatever depends on a heavy call is heavy, through calls, function values, requirements and branch joins alike. An instance call never yields for heaviness; a heavy instance's entry is its synchronous glue, called in place.

## 3. Where heavy work spawns

Heavy work spawns at each script call site whose callee is heavy, as early as its operands allow. Where to spawn is decided once, at the outermost heavy call, never inside an instance call. Work handed to the executor holds a loan only on storage the run keeps alive.

## 4. A run keeps its storage

A run keeps storage alive while any work of the run is aloft: work handed to the executor, and every callback run it started. Storage the run keeps is released only after its last such work has ended or been dropped (S3). "Storage the run keeps" in crossing means this.
