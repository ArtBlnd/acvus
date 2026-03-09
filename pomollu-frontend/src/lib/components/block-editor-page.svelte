<script lang="ts">
	import type { ContextBlock, RawBlock, ScriptBlock } from '$lib/types.js';
	import { blockKind, isContextBlock, isRawBlock, isScriptBlock } from '$lib/types.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import { createId, generateName, getOwnerChildren, updateOwnerBlock, collectScopeBlockNames, uiState } from '$lib/stores.svelte.js';
	import { findBlock } from '$lib/block-tree.js';
	import { onDestroy } from 'svelte';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import { Button } from '$lib/components/ui/button';
	import { X } from 'lucide-svelte';
	import BlockEditor from './block-editor.svelte';
	import RawBlockEditor from './raw-block-editor.svelte';
	import ScriptBlockEditor from './script-block-editor.svelte';

	let { blockId, owner, contextTypes = {} }: { blockId: string; owner: BlockOwner; contextTypes?: Record<string, import('$lib/type-parser.js').TypeDesc> } = $props();

	let block = $derived.by(() => {
		const children = getOwnerChildren(owner);
		return children ? findBlock(children, blockId) : undefined;
	});

	let scopeNames = $derived(collectScopeBlockNames(owner, blockId));
	let isDuplicate = $derived(block != null && block.name !== '' && scopeNames.includes(block.name));

	// Auto-fix duplicate name when component is destroyed (tab closed / block deselected)
	onDestroy(() => {
		const children = getOwnerChildren(owner);
		const b = children ? findBlock(children, blockId) : undefined;
		if (!b) return;
		const names = collectScopeBlockNames(owner, blockId);
		if (b.name !== '' && names.includes(b.name)) {
			const fixed = generateName(b.name, names);
			updateOwnerBlock(owner, blockId, (bl) => ({ ...bl, name: fixed }));
		}
	});

	function switchKind(kind: string) {
		if (!block) return;
		const current = blockKind(block);
		if (kind === current) return;

		const { id, name } = block;
		if (kind === 'none') {
			updateOwnerBlock(owner, blockId, () => ({ kind: 'none' as const, id, name }));
		} else if (kind === 'raw') {
			const raw: RawBlock = { kind: 'raw', id, name, text: '', mode: 'template' };
			updateOwnerBlock(owner, blockId, () => raw);
		} else if (kind === 'script') {
			const script: ScriptBlock = { kind: 'script', id, name, text: '' };
			updateOwnerBlock(owner, blockId, () => script);
		} else {
			const ctx: ContextBlock = {
				kind: 'context', id, name,
				info: { type: 'disabled', tags: {}, description: '' },
				priority: 0, content: [{ id: createId(), content_type: 'text', value: '' }], enabled: true
			};
			updateOwnerBlock(owner, blockId, () => ctx);
		}
	}

	function handleContextUpdate(updater: (b: ContextBlock) => ContextBlock) {
		updateOwnerBlock(owner, blockId, (b) => isContextBlock(b) ? updater(b) : b);
	}

	function handleRawUpdate(updater: (b: RawBlock) => RawBlock) {
		updateOwnerBlock(owner, blockId, (b) => isRawBlock(b) ? updater(b) : b);
	}

	function handleScriptUpdate(updater: (b: ScriptBlock) => ScriptBlock) {
		updateOwnerBlock(owner, blockId, (b) => isScriptBlock(b) ? updater(b) : b);
	}

	function handleRemove() {
		uiState.removeOwnerTreeNode(owner, blockId);
	}

	const kindLabels = { none: 'Disabled', context: 'Context', raw: 'Raw', script: 'Script' } as const;

	let locked = $derived(uiState.isOwnerBusy(owner));
</script>

<div class="flex h-full flex-col" class:pointer-events-none={locked} class:opacity-60={locked}>
	{#if locked}
		<div class="shrink-0 border-b bg-amber-500/10 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">Turn in progress — editing locked</div>
	{/if}
	{#if block}
		{@const kind = blockKind(block)}
		<div class="flex items-center gap-3 px-6 pt-10 pb-0" style="max-width: 42rem; margin: 0 auto; width: 100%;">
			<div class="flex-1 space-y-1">
				<Label>Name</Label>
				<Input
					value={block.name}
					oninput={(e) => updateOwnerBlock(owner, blockId, (b) => ({ ...b, name: e.currentTarget.value }))}
					placeholder="Block name..."
					class={isDuplicate ? 'border-destructive focus-visible:ring-destructive' : ''}
				/>
				{#if isDuplicate}
					<p class="text-xs text-destructive">Name is already used. Please choose a unique name.</p>
				{/if}
			</div>
			<div class="w-32 space-y-1">
				<Label>Kind</Label>
				<Select.Root type="single" value={kind} onValueChange={switchKind}>
					<Select.Trigger class="w-full">{kindLabels[kind]}</Select.Trigger>
					<Select.Content>
						<Select.Item value="none">Disabled</Select.Item>
						<Select.Item value="context">Context</Select.Item>
						<Select.Item value="raw">Raw</Select.Item>
						<Select.Item value="script">Script</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>
			{#if isRawBlock(block)}
				<div class="w-32 space-y-1">
					<Label>Mode</Label>
					<Select.Root type="single" value={block.mode} onValueChange={(v) => handleRawUpdate((b) => ({ ...b, mode: v as import('$lib/types.js').RawBlockMode }))}>
						<Select.Trigger class="w-full">{block.mode}</Select.Trigger>
						<Select.Content>
							<Select.Item value="template">template</Select.Item>
							<Select.Item value="script">script</Select.Item>
						</Select.Content>
					</Select.Root>
				</div>
			{/if}
			<div class="pt-5">
				<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive transition-colors" onclick={handleRemove}>
					<X class="h-4 w-4" />
				</Button>
			</div>
		</div>
		{#if isRawBlock(block)}
			<div class="flex flex-1 min-h-0 px-6 py-4">
				<RawBlockEditor block={block} onupdate={handleRawUpdate} {contextTypes} />
			</div>
		{:else}
			<ScrollArea class="flex-1">
				<div class="mx-auto max-w-2xl px-6 py-6">
					{#if isContextBlock(block)}
						<BlockEditor block={block} onupdate={handleContextUpdate} />
					{:else if isScriptBlock(block)}
						<ScriptBlockEditor block={block} onupdate={handleScriptUpdate} {contextTypes} />
					{:else}
						<div class="py-8 text-center text-sm text-muted-foreground">
							Select a kind to configure this block.
						</div>
					{/if}
				</div>
			</ScrollArea>
		{/if}
	{:else}
		<div class="p-4 text-sm text-muted-foreground">Block not found.</div>
	{/if}
</div>
