<script lang="ts">
	import * as Resizable from '$lib/components/ui/resizable';
	import BotSidebar from '$lib/components/bot-sidebar.svelte';
	import SessionSidebar from '$lib/components/session-sidebar.svelte';
	import SettingsSidebar from '$lib/components/settings-sidebar.svelte';
	import ContextSidebar from '$lib/components/context-sidebar.svelte';
	import ChatPanel from '$lib/components/chat-panel.svelte';
	import BotSettings from '$lib/components/bot-settings.svelte';
	import PromptSettings from '$lib/components/prompt-settings.svelte';
	import ProfileSettings from '$lib/components/profile-settings.svelte';
	import ProviderSettings from '$lib/components/provider-settings.svelte';
	import NodeSettings from '$lib/components/node-settings.svelte';
	import BlockEditorPage from '$lib/components/block-editor-page.svelte';
	import TabBar from '$lib/components/tab-bar.svelte';
	import SyncIndicator from '$lib/components/sync-indicator.svelte';
	import { sessionStore, botStore, promptStore, profileStore, providerStore, uiState } from '$lib/stores.svelte.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import { initPersistence } from '$lib/persistence.svelte.js';
	import { IndexedDBBackend } from '$lib/storage/indexeddb.js';
	import { analyzePrompt, analyzeProfile, analyzeBot } from '$lib/param-resolver.js';
	import type { ContextEnvResult, TwoPassResult } from '$lib/param-resolver.js';
	import type { ContextParam } from '$lib/types.js';
	import { onMount } from 'svelte';

	let activeTab = $derived(uiState.activeTab);

	// --- Owner env: inline discovery + orchestration typecheck ---

	const EMPTY_ENV: ContextEnvResult = { contextTypes: {}, nodeLocals: {}, nodeErrors: {} };
	const ENV_DEBOUNCE_MS = 300;

	let ownerEnv = $state<ContextEnvResult>(EMPTY_ENV);
	let ownerAnalysisErrors = $state<string[]>([]);
	let envTimer: ReturnType<typeof setTimeout> | null = null;
	let lastOwnerKey: string | null = null;

	function computeFullEnv(owner: BlockOwner): TwoPassResult {
		const getApi = (pid: string) => {
			const p = providerStore.get(pid);
			if (!p) throw new Error(`provider '${pid}' not found`);
			return p.api;
		};

		switch (owner.kind) {
			case 'prompt': {
				const prompt = promptStore.get(owner.promptId);
				if (!prompt) throw new Error(`prompt '${owner.promptId}' not found`);
				return analyzePrompt(prompt, getApi);
			}
			case 'profile': {
				const profile = profileStore.get(owner.profileId);
				if (!profile) throw new Error(`profile '${owner.profileId}' not found`);
				return analyzeProfile(profile, getApi);
			}
			case 'bot': {
				const bot = botStore.get(owner.botId);
				if (!bot) throw new Error(`bot '${owner.botId}' not found`);
				const prompt = promptStore.get(bot.promptId);
				if (!prompt) throw new Error(`prompt '${bot.promptId}' not found`);
				const profile = profileStore.get(bot.profileId);
				if (!profile) throw new Error(`profile '${bot.profileId}' not found`);
				return analyzeBot(bot, prompt, profile, getApi);
			}
		}
	}

	/** Sync params from twoPassAnalysis result into the owner's store. */
	function syncOwnerParams(owner: BlockOwner, params: ContextParam[]) {
		switch (owner.kind) {
			case 'prompt':
				promptStore.update(owner.promptId, (p) => ({ ...p, contextParams: params }));
				break;
			case 'profile':
				profileStore.update(owner.profileId, (p) => ({ ...p, contextParams: params }));
				break;
			case 'bot':
				botStore.update(owner.botId, (b) => ({ ...b, contextParams: params }));
				break;
		}
	}

	// Extract user-set fields only (resolution, userType) — NOT inferredType (which is an output).
	// This prevents a circular dependency: syncOwnerParams updates inferredType,
	// which must NOT re-trigger the env computation.
	function paramUserKey(params?: ContextParam[]) {
		return params?.map((p) => [p.name, p.resolution, p.userType]);
	}

	// Trigger key: captures all data that should cause a recomputation.
	let envTriggerKey = $derived.by(() => {
		if (!activeTab) return null;
		if (activeTab.kind !== 'block' && activeTab.kind !== 'node') return null;
		const owner = activeTab.owner;
		switch (owner.kind) {
			case 'prompt': {
				const p = promptStore.get(owner.promptId);
				return JSON.stringify(['prompt', owner.promptId, p?.contextBindings, p?.children, paramUserKey(p?.contextParams)]);
			}
			case 'profile': {
				const p = profileStore.get(owner.profileId);
				return JSON.stringify(['profile', owner.profileId, p?.children, paramUserKey(p?.contextParams)]);
			}
			case 'bot': {
				const bot = botStore.get(owner.botId);
				const p = promptStore.get(bot?.promptId ?? '');
				const pr = profileStore.get(bot?.profileId ?? '');
				return JSON.stringify(['bot', owner.botId,
					bot?.children, bot?.display, bot?.regions, paramUserKey(bot?.contextParams),
					p?.contextBindings, p?.children, paramUserKey(p?.contextParams),
					pr?.children, paramUserKey(pr?.contextParams),
				]);
			}
		}
	});

	// First load (tab switch) = immediate. Content changes = debounced. Both compute from scratch.
	function applyFullEnv(owner: BlockOwner) {
		const result = computeFullEnv(owner);
		if (!result.ok) {
			ownerEnv = EMPTY_ENV;
			ownerAnalysisErrors = result.errors;
			return;
		}
		ownerEnv = result.env;
		ownerAnalysisErrors = [];
		syncOwnerParams(owner, result.params);
	}

	$effect(() => {
		const key = envTriggerKey;

		if (key === null) {
			ownerEnv = EMPTY_ENV;
			lastOwnerKey = null;
			return;
		}
		if (!activeTab || (activeTab.kind !== 'block' && activeTab.kind !== 'node')) return;
		const owner = activeTab.owner;
		const ownerKey = JSON.stringify(owner);

		if (ownerKey !== lastOwnerKey) {
			lastOwnerKey = ownerKey;
			if (envTimer) clearTimeout(envTimer);
			applyFullEnv(owner);
		} else {
			if (envTimer) clearTimeout(envTimer);
			envTimer = setTimeout(() => {
				envTimer = null;
				applyFullEnv(owner);
			}, ENV_DEBOUNCE_MS);
		}

		return () => { if (envTimer) clearTimeout(envTimer); };
	});

	let cleanup: (() => void) | null = null;

	onMount(() => {
		IndexedDBBackend.open().then(async (backend) => {
			cleanup = await initPersistence(backend);
		});
		return () => cleanup?.();
	});

	$effect(() => {
		if (activeTab?.kind === 'chat') {
			sessionStore.activeSessionId = activeTab.sessionId;
		}
	});
</script>

<SyncIndicator />

<!-- Mobile toolbar -->
<div class="mobile-toolbar md:hidden flex items-center justify-between border-b bg-background px-3 py-2">
	<button
		class="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-accent"
		onclick={() => uiState.toggleLeftSidebar()}
	>
		<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
	</button>
	<span class="text-sm font-medium truncate">Pomollu</span>
	<button
		class="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-accent"
		onclick={() => uiState.toggleRightSidebar()}
	>
		<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18"/></svg>
	</button>
</div>

<!-- Mobile overlay backdrop -->
{#if uiState.leftSidebarOpen || uiState.rightSidebarOpen}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="mobile-backdrop md:hidden"
		onclick={() => uiState.closeMobileSidebars()}
		onkeydown={(e) => { if (e.key === 'Escape') uiState.closeMobileSidebars(); }}
	></div>
{/if}

<div class="flex h-screen flex-col md:flex-row bg-background text-foreground">
	<!-- Desktop: bot sidebar always visible -->
	<div class="hidden md:block">
		<BotSidebar />
	</div>

	<!-- Desktop: session/settings sidebar inline -->
	<div class="hidden md:block">
		{#if uiState.isSettings}
			<SettingsSidebar />
		{:else if uiState.selectedBotId}
			<SessionSidebar />
		{/if}
	</div>

	<!-- Mobile: left drawer (bot + session/settings) -->
	<div class="left-sidebar-panel md:hidden {uiState.leftSidebarOpen ? 'left-sidebar-open' : ''}">
		<div class="flex h-full">
			<BotSidebar />
			<div class="flex-1 min-w-0">
				{#if uiState.isSettings}
					<SettingsSidebar />
				{:else if uiState.selectedBotId}
					<SessionSidebar />
				{/if}
			</div>
		</div>
	</div>

	<!-- Main content area -->
	<div class="main-content flex-1 min-w-0 overflow-hidden md:contents">
		<Resizable.PaneGroup direction="horizontal" class="flex-1">
			<Resizable.Pane defaultSize={75} minSize={40}>
				<div class="flex h-full flex-col">
					{#if uiState.tabs.length > 0}
						<TabBar />
					{/if}
					<div class="flex-1 overflow-hidden">
						{#if activeTab?.kind === 'block'}
							<BlockEditorPage blockId={activeTab.blockId} owner={activeTab.owner} contextTypes={ownerEnv.contextTypes} analysisErrors={ownerAnalysisErrors} />
						{:else if activeTab?.kind === 'prompt'}
							<PromptSettings promptId={activeTab.promptId} />
						{:else if activeTab?.kind === 'profile'}
							<ProfileSettings profileId={activeTab.profileId} />
						{:else if activeTab?.kind === 'provider'}
							<ProviderSettings providerId={activeTab.providerId} />
						{:else if activeTab?.kind === 'node'}
							<NodeSettings nodeId={activeTab.nodeId} owner={activeTab.owner} contextTypes={ownerEnv.contextTypes} nodeLocals={ownerEnv.nodeLocals} nodeErrors={ownerEnv.nodeErrors} analysisErrors={ownerAnalysisErrors} />
						{:else if activeTab?.kind === 'bot-settings'}
							<BotSettings botId={activeTab.botId} />
						{:else if activeTab?.kind === 'chat'}
							{@const chatSession = sessionStore.sessions.find((s) => s.id === activeTab.sessionId)}
							{@const chatBot = chatSession ? botStore.get(chatSession.botId) : undefined}
							{#if chatSession && chatBot}
								<ChatPanel session={chatSession} bot={chatBot} />
							{:else}
								<div class="flex h-full items-center justify-center text-sm text-muted-foreground">Session or bot not found.</div>
							{/if}
						{:else}
							<div class="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
								<h1 class="text-2xl font-semibold text-foreground">Welcome to Pomollu!</h1>
								<a href="https://github.com/ArtBlnd/acvus" target="_blank" rel="noopener noreferrer" class="text-sm underline hover:text-foreground transition-colors">Source</a>
							</div>
						{/if}
					</div>
				</div>
			</Resizable.Pane>

			<Resizable.Handle withHandle class="hidden md:flex" />
			<Resizable.Pane defaultSize={25} minSize={10} maxSize={35} class="hidden md:block">
				<ContextSidebar />
			</Resizable.Pane>
		</Resizable.PaneGroup>
	</div>

	<!-- Right sidebar: overlay on mobile -->
	<div class="right-sidebar-panel md:hidden {uiState.rightSidebarOpen ? 'right-sidebar-open' : ''}">
		<ContextSidebar />
	</div>
</div>

<style>
	/* Mobile toolbar */
	.mobile-toolbar {
		position: sticky;
		top: 0;
		z-index: 40;
	}

	/* Mobile backdrop */
	.mobile-backdrop {
		position: fixed;
		inset: 0;
		z-index: 40;
		background: rgba(0, 0, 0, 0.4);
	}

	/* Main content on mobile */
	@media (max-width: 767px) {
		.main-content {
			display: flex;
			flex-direction: column;
			flex: 1;
			min-height: 0;
		}
	}

	/* Left sidebar: slide-in drawer on mobile */
	.left-sidebar-panel {
		position: fixed;
		top: 0;
		left: 0;
		bottom: 0;
		z-index: 50;
		width: 320px;
		max-width: 85vw;
		transform: translateX(-100%);
		transition: transform 0.25s ease;
		background: var(--color-background);
		box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
	}
	.left-sidebar-panel.left-sidebar-open {
		transform: translateX(0);
	}
	.left-sidebar-panel :global(> div > div:last-child > div) {
		width: 100% !important;
	}

	/* Right sidebar: slide-in drawer on mobile */
	.right-sidebar-panel {
		position: fixed;
		top: 0;
		right: 0;
		bottom: 0;
		z-index: 50;
		width: 300px;
		max-width: 85vw;
		transform: translateX(100%);
		transition: transform 0.25s ease;
		background: var(--color-background);
		box-shadow: -2px 0 8px rgba(0, 0, 0, 0.15);
	}
	.right-sidebar-panel.right-sidebar-open {
		transform: translateX(0);
	}
</style>
