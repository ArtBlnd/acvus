<script lang="ts">
	import type { TurnNode } from '$lib/engine.js';
	import { tick } from 'svelte';

	let {
		nodes,
		cursor,
		onGoto,
		disabled = false,
	}: {
		nodes: TurnNode[];
		cursor: string;
		onGoto: (id: string) => void;
		disabled?: boolean;
	} = $props();

	// Build parent→children map
	let childrenMap = $derived.by(() => {
		const map = new Map<string, string[]>();
		for (const node of nodes) {
			const parentKey = node.parent != null ? node.parent : '__root__';
			const children = map.get(parentKey);
			if (children) {
				children.push(node.uuid);
			} else {
				map.set(parentKey, [node.uuid]);
			}
		}
		return map;
	});

	// Build uuid→node lookup
	let nodeMap = $derived.by(() => {
		const map = new Map<string, TurnNode>();
		for (const node of nodes) {
			map.set(node.uuid, node);
		}
		return map;
	});

	// Ancestors of cursor (path from cursor back to root)
	let ancestorSet = $derived.by(() => {
		const set = new Set<string>();
		let current: string | null = cursor;
		while (current) {
			set.add(current);
			const node = nodeMap.get(current);
			current = node?.parent ?? null;
		}
		return set;
	});

	type TreeRow = {
		node: TurnNode;
		column: number;
		isAncestor: boolean;
		isCursor: boolean;
		isRoot: boolean;
		parentColumn: number | null;
	};

	let rows = $derived.by(() => {
		const result: TreeRow[] = [];
		const visited = new Set<string>();
		const nodeColumnMap = new Map<string, number>();

		function walk(uuid: string, col: number) {
			if (visited.has(uuid)) return;
			visited.add(uuid);

			const node = nodeMap.get(uuid);
			if (!node) return;

			nodeColumnMap.set(uuid, col);
			const parentCol = node.parent ? (nodeColumnMap.get(node.parent) ?? null) : null;

			result.push({
				node,
				column: col,
				isAncestor: ancestorSet.has(uuid),
				isCursor: uuid === cursor,
				isRoot: node.parent == null,
				parentColumn: parentCol,
			});

			const children = childrenMap.get(uuid) ?? [];
			// First child continues on the same column (ancestor preferred),
			// additional children branch out to col + 1.
			const firstChild = children.find((c) => ancestorSet.has(c)) ?? children[0];
			const rest = children.filter((c) => c !== firstChild);

			if (firstChild) {
				walk(firstChild, col);
			}
			for (const branchId of rest) {
				walk(branchId, col + 1);
			}
		}

		const roots = nodes.filter((n) => n.parent == null);
		for (const root of roots) {
			walk(root.uuid, 0);
		}

		return result;
	});

	// Compute active columns per row for vertical line drawing
	let graphData = $derived.by(() => {
		const maxCol = rows.reduce((m, r) => Math.max(m, r.column), 0);
		// Find last row index for each column
		const lastRowForCol = new Map<number, number>();
		for (let i = 0; i < rows.length; i++) {
			lastRowForCol.set(rows[i].column, i);
		}
		// Forward pass: track active columns
		const active = new Set<number>();
		const perRow: { activeColumns: Set<number> }[] = [];
		for (let i = 0; i < rows.length; i++) {
			const col = rows[i].column;
			active.add(col);
			perRow.push({ activeColumns: new Set(active) });
			if (lastRowForCol.get(col) === i) {
				active.delete(col);
			}
		}
		return { maxCol, perRow };
	});

	const COL_W = 18;
	const ROW_H = 28;
	const DOT_R = 4;

	let scrollEl: HTMLDivElement;

	// Scroll to bottom when nodes change
	$effect(() => {
		void nodes.length;
		void cursor;
		tick().then(() => {
			if (scrollEl) {
				scrollEl.scrollTop = scrollEl.scrollHeight;
			}
		});
	});
</script>

<div class="flex h-full flex-col bg-background">
	<div class="flex items-center gap-1.5 border-b px-3 py-2">
		<span class="text-xs font-medium">Tree</span>
		<span class="text-[10px] text-muted-foreground">{nodes.length} nodes</span>
	</div>

	<div class="flex-1 overflow-y-auto" bind:this={scrollEl}>
		<div class="py-1 px-2">
			{#each rows as row, i (row.node.uuid)}
				{@const gd = graphData.perRow[i]}
				{@const graphW = (graphData.maxCol + 1) * COL_W}
				{@const cx = row.column * COL_W + COL_W / 2}
				<div class="flex items-center" style="height: {ROW_H}px;">
					<!-- Graph area -->
					<svg
						width={graphW}
						height={ROW_H}
						class="shrink-0"
						style="min-width: {graphW}px;"
					>
						<!-- Vertical lines for active columns -->
						{#each [...gd.activeColumns] as col}
							{@const lx = col * COL_W + COL_W / 2}
							{#if col !== row.column}
								<!-- Full vertical line (pass-through) -->
								<line
									x1={lx} y1={0} x2={lx} y2={ROW_H}
									stroke="var(--color-muted-foreground)"
									stroke-opacity="0.25"
									stroke-width="1.5"
								/>
							{:else}
								<!-- Vertical line above dot -->
								{#if !row.isRoot}
									<line
										x1={lx} y1={0} x2={lx} y2={ROW_H / 2 - DOT_R}
										stroke="var(--color-muted-foreground)"
										stroke-opacity={row.isAncestor ? "0.5" : "0.25"}
										stroke-width="1.5"
									/>
								{/if}
								<!-- Vertical line below dot (if not the last on this column) -->
								{#if i < rows.length - 1 || (childrenMap.get(row.node.uuid)?.length ?? 0) > 0}
									{@const isLastOnCol = !rows.slice(i + 1).some((r) => r.column === col)}
									{#if !isLastOnCol}
										<line
											x1={lx} y1={ROW_H / 2 + DOT_R} x2={lx} y2={ROW_H}
											stroke="var(--color-muted-foreground)"
											stroke-opacity={row.isAncestor ? "0.5" : "0.25"}
											stroke-width="1.5"
										/>
									{/if}
								{/if}
							{/if}
						{/each}

						<!-- Horizontal connector from parent column to this column -->
						{#if row.parentColumn != null && row.parentColumn !== row.column}
							{@const px = row.parentColumn * COL_W + COL_W / 2}
							<path
								d="M {px} {0} L {px} {ROW_H / 2} L {cx - DOT_R} {ROW_H / 2}"
								fill="none"
								stroke="var(--color-muted-foreground)"
								stroke-opacity="0.35"
								stroke-width="1.5"
							/>
						{/if}

						<!-- Node dot -->
						{#if row.isCursor}
							<circle cx={cx} cy={ROW_H / 2} r={DOT_R} fill="var(--color-primary)" />
						{:else if row.isAncestor}
							<circle cx={cx} cy={ROW_H / 2} r={DOT_R - 0.5} fill="var(--color-foreground)" opacity="0.7" />
						{:else if row.isRoot}
							<circle cx={cx} cy={ROW_H / 2} r={DOT_R - 0.5} fill="none" stroke="var(--color-muted-foreground)" stroke-width="1.5" opacity="0.5" />
						{:else}
							<circle cx={cx} cy={ROW_H / 2} r={DOT_R - 1} fill="none" stroke="var(--color-muted-foreground)" stroke-width="1" opacity="0.4" />
						{/if}
					</svg>

					<!-- Label -->
					<button
						class="group flex items-center gap-1.5 flex-1 min-w-0 text-left rounded px-1.5 h-full transition-colors
							{row.isCursor ? 'bg-primary/10' : disabled ? '' : 'hover:bg-accent/50'}
							{disabled ? 'opacity-50 cursor-not-allowed' : ''}"
						onclick={() => onGoto(row.node.uuid)}
						{disabled}
						title={row.node.uuid}
					>
						<span class="text-[11px] truncate {row.isCursor ? 'text-primary font-medium' : row.isAncestor ? 'text-foreground' : 'text-muted-foreground'}">
							{#if row.isRoot}
								Root
							{:else}
								Turn {row.node.depth}
							{/if}
						</span>
						<code class="text-[9px] text-muted-foreground/60 font-mono ml-auto shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
							{row.node.uuid.slice(0, 7)}
						</code>
					</button>
				</div>
			{/each}

			{#if nodes.length === 0}
				<div class="px-2 py-4 text-center text-xs text-muted-foreground">
					No history yet.
				</div>
			{/if}
		</div>
	</div>
</div>
