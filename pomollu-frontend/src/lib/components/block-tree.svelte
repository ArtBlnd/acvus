<script lang="ts">
	import type { BlockNode } from '$lib/types.js';
	import { isContextBlock, isRawBlock, isScriptBlock } from '$lib/types.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import { uiState, getOwnerChildren, updateOwnerChildren } from '$lib/stores.svelte.js';
	import { nodeId, DROP_PREFIX, updateTreeNode } from '$lib/block-tree.js';
	import { longpress } from '$lib/actions/longpress.js';
	import DragItem from './dnd/drag-item.svelte';
	import DropZone from './dnd/drop-zone.svelte';
	import { tick } from 'svelte';

	let {
		children,
		owner,
		dragSourceId = null,
		dropTargetId = null,
		collapsedIds,
		ontogglefolder,
		depth = 0
	}: {
		children: BlockNode[];
		owner: BlockOwner;
		dragSourceId?: string | null;
		dropTargetId?: string | null;
		collapsedIds: Set<string>;
		ontogglefolder: (id: string) => void;
		depth?: number;
	} = $props();

	function dropId(id: string) { return DROP_PREFIX + id; }

	function deleteItem(e: MouseEvent, id: string) {
		e.stopPropagation();
		uiState.removeOwnerTreeNode(owner, id);
	}

	// --- Folder rename ---
	let editingFolderId = $state<string | null>(null);
	let editingName = $state('');

	function startRename(id: string, currentName: string) {
		editingFolderId = id;
		editingName = currentName;
	}

	function commitRename() {
		if (!editingFolderId) return;
		const id = editingFolderId;
		const name = editingName.trim();
		editingFolderId = null;
		if (!name) return;
		const tree = getOwnerChildren(owner);
		if (!tree) return;
		updateOwnerChildren(owner, updateTreeNode(tree, id, (n) => {
			if (n.kind !== 'folder') return n;
			return { kind: 'folder', folder: { ...n.folder, name } };
		}));
	}

	function cancelRename() {
		editingFolderId = null;
	}

	function handleFolderAuxClick(e: MouseEvent, id: string, name: string) {
		if (e.button === 1) {
			e.preventDefault();
			startRename(id, name);
		}
	}

	function handleFolderLongPress(id: string, name: string) {
		startRename(id, name);
	}
</script>

{#snippet containerBody(id: string, childNodes: BlockNode[])}
	{@const collapsed = collapsedIds.has(id)}
	{#if !collapsed}
		{#if childNodes.length > 0}
			<div class="folder-body">
				<svelte:self
					children={childNodes}
					{owner}
					{dragSourceId}
					{dropTargetId}
					{collapsedIds}
					{ontogglefolder}
					depth={depth + 1}
				/>
				<DropZone id={dropId(id)} active={dropTargetId === dropId(id)} />
			</div>
		{:else}
			<DropZone id={dropId(id)} active={dropTargetId === dropId(id)}>
				drop here
			</DropZone>
		{/if}
	{/if}
{/snippet}

<div class="tree-list">
	{#each children as node (nodeId(node))}
		{#if node.kind === 'block'}
			<!-- svelte-ignore a11y_click_events_have_key_events -->
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div class="tree-item" use:longpress>
				<div onclick={() => uiState.openBlock(node.block.id, owner)}>
					<DragItem
						id={node.block.id}
						isDragSource={dragSourceId === node.block.id}
						isDropTarget={dropTargetId === node.block.id}
						style="background: var(--color-card); border: 1px solid var(--color-border);"
					>
						{#if isContextBlock(node.block)}
							{#if node.block.info.type === 'disabled'}
								<span class="badge badge-raw">disabled</span>
							{:else}
								<span class="badge">{node.block.info.type}</span>
							{/if}
							<span class="item-name" class:item-disabled={node.block.info.type === 'disabled'}>{node.block.name || node.block.info.description || 'Untitled'}</span>
						{:else if isRawBlock(node.block)}
							<span class="badge badge-raw">raw</span>
							<span class="item-name">{node.block.name || 'Untitled'}</span>
						{:else if isScriptBlock(node.block)}
							<span class="badge badge-script">script</span>
							<span class="item-name">{node.block.name || 'Untitled'}</span>
						{:else}
							<span class="item-name item-disabled">{node.block.name || 'Untitled'}</span>
						{/if}
					</DragItem>
				</div>
				<button class="item-delete" onclick={(e) => deleteItem(e, node.block.id)} title="Delete">&times;</button>
			</div>
		{:else if node.kind === 'node'}
			<!-- svelte-ignore a11y_click_events_have_key_events -->
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<div class="tree-item" use:longpress>
				<div onclick={() => uiState.openNode(node.node.id, owner)}>
					<DragItem
						id={node.node.id}
						isDragSource={dragSourceId === node.node.id}
						isDropTarget={dropTargetId === node.node.id}
						style="background: var(--color-card); border: 1px solid var(--color-border);
							{uiState.isNodeOpen(node.node.id) ? 'border-color: var(--color-primary);' : ''}"
					>
						<span class="badge badge-sm">{node.node.kind}</span>
						<span class="item-name">{node.node.name}</span>
					</DragItem>
				</div>
				<button class="item-delete" onclick={(e) => deleteItem(e, node.node.id)} title="Delete">&times;</button>
			</div>
		{:else}
			<div class="container-wrapper">
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="tree-item"
					use:longpress={{ onlongpress: () => handleFolderLongPress(node.folder.id, node.folder.name) }}
					onauxclick={(e) => handleFolderAuxClick(e, node.folder.id, node.folder.name)}
				>
					<div onclick={() => { if (editingFolderId !== node.folder.id) ontogglefolder(node.folder.id); }}>
						<DragItem
							id={node.folder.id}
							isDragSource={dragSourceId === node.folder.id}
							isDropTarget={dropTargetId === node.folder.id}
							style="background: var(--color-secondary); border: 1px solid var(--color-border); font-weight: 500; font-size: 13px;"
						>
							{#if editingFolderId === node.folder.id}
								<!-- svelte-ignore a11y_autofocus -->
								<input
									class="folder-rename-input"
									value={editingName}
									oninput={(e) => editingName = e.currentTarget.value}
									onkeydown={(e) => { if (e.key === 'Enter') commitRename(); if (e.key === 'Escape') cancelRename(); }}
									onblur={commitRename}
									onclick={(e) => e.stopPropagation()}
									onmousedown={(e) => e.stopPropagation()}
									autofocus
								/>
							{:else}
								{node.folder.name}
								{#if collapsedIds.has(node.folder.id) && node.folder.children.length > 0}
									<span class="child-count">{node.folder.children.length}</span>
								{/if}
							{/if}
						</DragItem>
					</div>
					<button class="item-delete" onclick={(e) => deleteItem(e, node.folder.id)} title="Delete">&times;</button>
				</div>
				{@render containerBody(node.folder.id, node.folder.children)}
			</div>
		{/if}
	{/each}
</div>

<style>
	.tree-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.container-wrapper {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.folder-body {
		margin-left: 8px;
		border-left: 2px solid var(--color-border);
		padding-left: 8px;
	}
	:global(.drag-item) {
		padding: 10px 12px;
		border-radius: 6px;
		cursor: grab;
		touch-action: none;
		font-size: 13px;
	}
	.badge {
		flex-shrink: 0;
		padding: 2px 6px;
		border-radius: 4px;
		background-color: var(--color-primary);
		color: var(--color-primary-foreground);
		font-size: 11px;
		font-weight: 500;
		margin-right: 8px;
	}
	.badge-sm {
		font-size: 10px;
	}
	.item-disabled {
		opacity: 0.4;
	}
	.badge-raw {
		background-color: var(--color-muted);
		color: var(--color-muted-foreground);
	}
	.badge-tool {
		background-color: hsl(30 80% 50%);
		color: white;
	}
	.badge-script {
		background-color: hsl(260 60% 55%);
		color: white;
	}
	.item-name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.child-count {
		margin-left: auto;
		font-size: 10px;
		opacity: 0.5;
		font-weight: 400;
	}

	/* Delete button */
	.tree-item {
		position: relative;
	}
	.item-delete {
		position: absolute;
		right: 6px;
		top: 50%;
		transform: translateY(-50%);
		opacity: 0;
		pointer-events: none;
		transition: opacity 0.15s;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 20px;
		height: 20px;
		border-radius: 4px;
		border: none;
		background: var(--color-destructive);
		color: var(--color-destructive-foreground);
		font-size: 14px;
		line-height: 1;
		cursor: pointer;
		z-index: 1;
	}
	@media (hover: hover) {
		.tree-item:hover > .item-delete {
			opacity: 1;
			pointer-events: auto;
		}
	}
	:global([data-long-pressed] > .item-delete) {
		opacity: 1;
		pointer-events: auto;
	}
	.folder-rename-input {
		flex: 1;
		min-width: 0;
		background: transparent;
		border: none;
		border-bottom: 1px solid var(--color-primary);
		outline: none;
		color: var(--color-foreground);
		font: inherit;
		padding: 0;
	}
</style>
