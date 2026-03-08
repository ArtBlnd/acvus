<script lang="ts">
	import type { Tab } from '$lib/stores.svelte.js';
	import { blockLabel } from '$lib/types.js';
	import { uiState, botStore, promptStore, profileStore, providerStore, sessionStore, getOwnerChildren, tabKey } from '$lib/stores.svelte.js';
	import { findBlock, findNodeItem } from '$lib/block-tree.js';
	import { longpress } from '$lib/actions/longpress.js';
	import { X } from 'lucide-svelte';
	import { slide } from 'svelte/transition';

	function tabLabel(tab: Tab): string {
		switch (tab.kind) {
			case 'bot-settings': {
				const bot = botStore.get(tab.botId);
				return bot ? `${bot.name} Settings` : 'Bot Settings';
			}
			case 'chat': {
				const session = sessionStore.sessions.find((s) => s.id === tab.sessionId);
				return session?.name || 'Chat';
			}
			case 'block': {
				const children = getOwnerChildren(tab.owner);
				if (!children) return 'Block';
				const block = findBlock(children, tab.blockId);
				if (!block) return 'Block';
				return blockLabel(block);
			}
			case 'prompt':
				return promptStore.get(tab.promptId)?.name || 'Prompt';
			case 'profile':
				return profileStore.get(tab.profileId)?.name || 'Profile';
			case 'node': {
				const nodeChildren = getOwnerChildren(tab.owner);
				if (!nodeChildren) return 'Node';
				return findNodeItem(nodeChildren, tab.nodeId)?.name || 'Node';
			}
			case 'provider':
				return providerStore.get(tab.providerId)?.name || 'Provider';
		}
	}

	function handleMouseDown(e: MouseEvent, index: number) {
		if (e.button === 1) {
			e.preventDefault();
			uiState.closeTab(index);
		}
	}
</script>

<div class="tab-bar flex shrink-0 items-end gap-0 overflow-x-auto border-b bg-background">
	{#each uiState.tabs as tab, i (tabKey(tab))}
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="tab-item group flex max-w-48 cursor-pointer items-center gap-1 border-r px-4 py-2 text-sm transition-colors select-none min-w-[120px]
				{i === uiState.activeTabIndex
				? 'border-b-2 border-b-primary bg-background text-foreground shadow-[inset_0_2px_0_0_hsl(var(--primary))]'
				: 'text-muted-foreground hover:bg-accent/50'}"
			role="tab"
			tabindex="0"
			aria-selected={i === uiState.activeTabIndex}
			onclick={() => (uiState.activeTabIndex = i)}
			onmousedown={(e) => handleMouseDown(e, i)}
			onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') uiState.activeTabIndex = i; }}
			transition:slide={{ duration: 200, axis: 'x' }}
			use:longpress
		>
			<span class="min-w-0 flex-1 truncate">{tabLabel(tab)}</span>
			<button
				class="tab-close ml-2 shrink-0 rounded-sm p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground
					{i === uiState.activeTabIndex ? 'desktop-visible' : ''}"
				onclick={(e) => { e.stopPropagation(); uiState.closeTab(i); }}
				aria-label="Close tab"
			>
				<X class="h-3.5 w-3.5" />
			</button>
		</div>
	{/each}
</div>

<style>
	.tab-bar {
		scrollbar-width: none;
	}
	.tab-bar::-webkit-scrollbar {
		display: none;
	}
	.tab-close {
		opacity: 0;
		pointer-events: none;
		transition: opacity 0.15s;
	}
	/* Desktop: hover or active tab */
	@media (hover: hover) {
		.tab-item:hover > .tab-close {
			opacity: 1;
			pointer-events: auto;
		}
		.tab-close.desktop-visible {
			opacity: 0.5;
			pointer-events: auto;
		}
	}
	/* Touch devices: always show close on active tab */
	@media (hover: none) {
		.tab-close.desktop-visible {
			opacity: 0.5;
			pointer-events: auto;
		}
	}
	/* Touch: long-press */
	:global([data-long-pressed] > .tab-close) {
		opacity: 1;
		pointer-events: auto;
	}
</style>
