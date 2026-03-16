<script lang="ts">
	import type { ContextBlock, RawBlock } from '$lib/types.js';
	import { blockKind, isContextBlock, isRawBlock } from '$lib/types.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import type { DocumentManager } from '$lib/document-manager.svelte.js';
	import { createId, generateName, getOwnerChildren, updateOwnerBlock, collectScopeBlockNames, uiState } from '$lib/stores.svelte.js';
	import { collectOwnerDeps } from '$lib/dependencies.js';
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

	import BasePage from './base-page.svelte';

	let {
		blockId,
		owner,
		contextTypes = {},
		parentLocals,
		docManager,
		level,
	}: {
		blockId: string;
		owner: BlockOwner;
		contextTypes?: Record<string, import('$lib/type-parser.js').TypeDesc>;
		parentLocals?: { raw: import('$lib/type-parser.js').TypeDesc; self: import('$lib/type-parser.js').TypeDesc };
		docManager?: DocumentManager;
		level?: string;
	} = $props();

	let mergedContextTypes = $derived.by((): Record<string, import('$lib/type-parser.js').TypeDesc> => {
		if (!parentLocals) return contextTypes;
		return { ...contextTypes, raw: parentLocals.raw, self: parentLocals.self };
	});

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

	function handleRemove() {
		uiState.removeOwnerTreeNode(owner, blockId);
	}

	const kindLabels = { none: 'Disabled', context: 'Context', raw: 'Raw' } as const;

	let deps = $derived(collectOwnerDeps(owner));
</script>

<BasePage {deps} onConfigChange={() => {}}>
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
				<RawBlockEditor block={block} onupdate={handleRawUpdate} {docManager} docKey={docManager && level ? `${level}:rawblock:${block.id}` : undefined} />
			</div>
		{:else}
			<ScrollArea class="flex-1">
				<div class="mx-auto max-w-2xl px-6 py-6">
					{#if isContextBlock(block)}
						<BlockEditor block={block} onupdate={handleContextUpdate} />
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
</BasePage>
