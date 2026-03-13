<script lang="ts">
	import type { BlockNode } from '$lib/types.js';
	import { blockLabel } from '$lib/types.js';
	import { botStore, sessionStore, uiState, generateName, createBlockNode, createFolderNode, createNodeBlockNode, getOwnerChildren, updateOwnerChildren, addOwnerChild } from '$lib/stores.svelte.js';
	import { collectAllNames } from '$lib/block-tree.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import BlockListCompact from './block-list-compact.svelte';
	import HistoryPanel from './history-panel.svelte';

	type OwnerContext = { kind: 'owner'; owner: BlockOwner; children: BlockNode[]; label: string };

	const labels: Record<BlockOwner['kind'], string> = { bot: 'BOT', prompt: 'PROMPT', profile: 'PROFILE' };
	const defaultBlockTypes: Record<BlockOwner['kind'], string> = { profile: 'memory', bot: 'character', prompt: 'system' };

	function ownerContext(owner: BlockOwner): OwnerContext | null {
		const children = getOwnerChildren(owner);
		if (!children) return null;
		return { kind: 'owner', owner, children, label: labels[owner.kind] };
	}

	function tabOwner(): BlockOwner | null {
		const tab = uiState.activeTab;
		if (!tab) return null;
		if (tab.kind === 'node' || tab.kind === 'block') return tab.owner;
		if (tab.kind === 'prompt') return { kind: 'prompt', promptId: tab.promptId };
		if (tab.kind === 'profile') return { kind: 'profile', profileId: tab.profileId };
		return null;
	}

	let chatState = $derived(
		sessionStore.activeSessionId
			? sessionStore.getChatState(sessionStore.activeSessionId)
			: null
	);

	let context = $derived.by(() => {
		const tab = uiState.activeTab;
		if (tab?.kind === 'chat') return { kind: 'session' as const };
		const owner = tabOwner();
		if (owner) return ownerContext(owner);
		if (tab?.kind === 'bot-settings') {
			const bot = botStore.active;
			if (bot) return ownerContext({ kind: 'bot', botId: bot.id });
		}
		return null;
	});
</script>

<div class="flex h-full flex-col">
	{#if context?.kind === 'session' && chatState}
		<HistoryPanel
			nodes={chatState.treeNodes}
			cursor={chatState.treeCursor}
			onGoto={(id) => chatState.gotoHandler?.(id)}
			disabled={chatState.isLoading}
		/>
	{:else if context?.kind === 'owner'}
		<div class="shrink-0 border-b px-3 py-2.5">
			<div class="text-xs font-medium text-muted-foreground">{context.label}</div>
		</div>
		<div class="flex-1 overflow-x-hidden overflow-y-auto p-2">
			<BlockListCompact
				children={context.children}
				owner={context.owner}
				onchildrenchange={(c) => updateOwnerChildren(context.owner, c)}
				onadd={() => addOwnerChild(context.owner, createBlockNode())}
				onaddfolder={() => addOwnerChild(context.owner, createFolderNode(generateName('New Folder', collectAllNames(context.children))))}
				onaddnode={() => addOwnerChild(context.owner, createNodeBlockNode(generateName('New Node', collectAllNames(context.children))))}
			/>
		</div>
	{/if}
</div>
