<script lang="ts">
	import type { RenderedRegion, RenderedCard, GridLayout, Session, Bot } from '$lib/types.js';
	import { GRID_HISTORY, HISTORY_BINDING_NAME, gridStyle, computePlacements } from '$lib/types.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Button } from '$lib/components/ui/button';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Label } from '$lib/components/ui/label';
	import { Send, MessageSquarePlus, Loader2, Square, RotateCcw } from 'lucide-svelte';
	import { tick } from 'svelte';
	import DisplayCard from './display-card.svelte';
	import { sessionStore, promptStore, profileStore, uiState } from '$lib/stores.svelte.js';
	import { ChatSession, type TurnNode } from '$lib/engine.js';
	import { buildSessionConfig, type BuildResult } from '$lib/session-builder.js';
	import { confirmAction } from '$lib/confirm-dialog.svelte.js';
	import { collectBotDeps } from '$lib/dependencies.js';
	import type { EntityRef } from '$lib/entity-versions.svelte.js';
	import BasePage from './base-page.svelte';

	let {
		session,
		bot,
	}: {
		session: Session;
		bot: Bot;
	} = $props();

	// --- Ephemeral state: survives component remounts ---

	const st = sessionStore.getChatState(session.id);
	let inputValue = $state('');
	let prevSessionKey = '';

	// --- Derived from props (always recomputed, not ephemeral) ---

	let prompt = $derived(promptStore.get(bot.promptId));

	let deps = $derived.by(() => {
		const p = promptStore.get(bot.promptId);
		const pr = profileStore.get(bot.profileId);
		if (!p || !pr) return [{ kind: 'bot' as const, id: bot.id }];
		return collectBotDeps(bot, p, pr);
	});

	let inputParam = $derived(prompt?.inputParam);
	let hasHistoryBinding = $derived(
		prompt?.contextBindings.some((b) => b.name === HISTORY_BINDING_NAME && b.script.trim() !== '') ?? false
	);
	let isConfigured = $derived(hasHistoryBinding && !!inputParam);
	let canSubmit = $derived(inputValue.trim().length > 0 && isConfigured);
	let canReroll = $derived(!st.isLoading && st.turnCount > 0 && isConfigured && st.lastInput !== null);

	// --- Error handling: config built via debounced handleConfigChange, runtime errors are ephemeral ---
	let configResult = $state<BuildResult | null>(buildSessionConfig(bot));
	let configError = $derived(
		!configResult ? 'No nodes configured.'
		: !configResult.ok ? configResult.errors.join('\n')
		: ''
	);
	let errorMsg = $derived(configError || st.runtimeError);

	// Whether display engine is active
	let useDisplayEngine = $derived(
		bot.display.iterator.trim() !== '' && bot.display.entries.length > 0
	);

	// --- Component-local state (OK to lose on remount) ---
	let loadingMore = $state(false);
	let sentinelEl = $state<HTMLElement>();
	let viewportEl = $state<HTMLElement | null>(null);


	// --- Session lifecycle: auto-init when conditions are met ---
	$effect(() => {
		const key = `${bot.id}:${session.id}`;
		if (key !== prevSessionKey) {
			prevSessionKey = key;

			// Abort pending resolver so the old turn() can reject and clean up
			if (st.pendingResolve) {
				st.pendingResolve.resolve('');
				st.pendingResolve = null;
				st.pendingValue = '';
			}

			// Clear stale display state
			didInitialScroll = false;
			st.displayCards = [];
			st.totalListLen = 0;
			st.loadedFrom = 0;
			st.regionData = new Map();
			st.runtimeError = '';
			st.turnCount = 0;
			st.treeNodes = [];
			st.treeCursor = '';

			// Invalidate old session
			if (st.chatSession) {
				st.chatSession.free();
				st.chatSession = null;
				st.chatSessionKey = null;
			}
		}
	});

	// Called by BasePage when any config store mutates (debounced).
	function handleConfigChange() {
		if (st.isLoading) return;
		// Rebuild config (expensive — only runs debounced via BasePage).
		configResult = buildSessionConfig(bot);
		// Invalidate old session — will be rebuilt with fresh config.
		if (st.chatSession) {
			st.chatSession.free();
			st.chatSession = null;
			st.chatSessionKey = null;
		}
		st.runtimeError = '';
	}

	// First init: when session key changes and config is valid.
	$effect(() => {
		if (st.chatSession || configError || !isConfigured || st.isLoading) return;
		st.runtimeError = '';
		ensureChatSession().catch((e) => {
			st.runtimeError = e instanceof Error ? e.message : String(e);
		});
	});

	async function ensureChatSession(): Promise<ChatSession> {
		const key = `${bot.id}:${session.id}`;
		if (st.chatSession && st.chatSessionKey === key) return st.chatSession;
		if (st.chatSession) {
			st.chatSession.free();
			st.chatSession = null;
			st.chatSessionKey = null;
		}

		if (!configResult?.ok) throw new Error(configError || 'invalid config');

		const cs = await ChatSession.create(configResult.config, session.id);

		// Guard: another init may have completed while we were awaiting
		if (st.chatSessionKey !== null) {
			cs.free();
			return st.chatSession!;
		}

		st.chatSession = cs;
		st.chatSessionKey = key;
		st.gotoHandler = (id: string) => handleHistoryGoto(id);

		st.turnCount = await cs.turnCount();

		const treeView = cs.tree();
		if (treeView) {
			st.treeNodes = treeView.nodes;
			st.treeCursor = treeView.cursor;
		}

		if (useDisplayEngine) {
			await initialDisplayLoad(cs);
		}

		return cs;
	}

	async function initialDisplayLoad(cs: ChatSession) {
		if (cs.freed) return;
		const len = await cs.displayListLen(bot.display.iterator);
		st.totalListLen = len;
		st.loadedFrom = Math.max(0, len - 5);
		if (len > 0) {
			st.displayCards = await renderDisplayRange(cs, st.loadedFrom, len);
		} else {
			st.displayCards = [];
		}
		scrollToBottom();
	}

	// Re-render static regions when bot.regions changes
	let regionsKey = $derived(JSON.stringify(bot.regions));
	$effect(() => {
		void regionsKey;
		if (!isConfigured || st.isLoading) return;
		ensureChatSession().then((cs) => {
			if (!st.isLoading) renderRegions(cs);
		}).catch((e) => {
			st.runtimeError = e instanceof Error ? e.message : String(e);
		});
	});

	// Layout from bot
	let layout = $derived<GridLayout>(bot.layout);
	let placements = $derived(computePlacements(layout));
	let hasRegions = $derived(placements.some((p) => p.id !== GRID_HISTORY));
	let hasAspect = $derived((layout.aspect ?? 0) > 0);
	let computedGridStyle = $derived(gridStyle(layout));

	function scrollToBottom() {
		if (viewportEl) {
			viewportEl.scrollTop = viewportEl.scrollHeight;
		}
	}

	// When viewport mounts and cards exist, scroll to bottom once.
	let didInitialScroll = false;
	$effect(() => {
		if (viewportEl && st.displayCards.length > 0 && !didInitialScroll) {
			didInitialScroll = true;
			viewportEl.scrollTop = viewportEl.scrollHeight;
		}
	});

	function formatResult(value: unknown): string {
		if (typeof value === 'string') return value;
		if (value && typeof value === 'object' && 'content' in value && typeof (value as Record<string, unknown>).content === 'string') {
			return (value as Record<string, string>).content;
		}
		return JSON.stringify(value);
	}

	// --- Display engine helpers ---

	function buildEntriesJson(): string {
		return JSON.stringify(
			bot.display.entries.map((e) => ({
				name: e.name,
				condition: e.condition,
				template: e.template,
			}))
		);
	}

	async function renderDisplayRange(cs: ChatSession, start: number, end: number): Promise<RenderedCard[]> {
		const entriesJson = buildEntriesJson();
		const cards: RenderedCard[] = [];
		for (let i = start; i < end; i++) {
			const rendered = await cs.renderDisplay(bot.display.iterator, entriesJson, i);
			cards.push(...rendered);
		}
		return cards;
	}

	async function renderRegions(cs: ChatSession) {
		const newRegionData = new Map(st.regionData);
		for (const region of bot.regions) {
			try {
				if (region.kind === 'static') {
					const cards = await cs.renderStatic(region.template);
					newRegionData.set(region.id, { id: region.id, name: region.name, cards });
				} else if (region.kind === 'iterable') {
					const len = await cs.displayListLen(region.iterator);
					const entriesJson = JSON.stringify(
						region.entries.map((e) => ({
							name: e.name,
							condition: e.condition,
							template: e.template,
						}))
					);
					const cards: RenderedCard[] = [];
					for (let i = 0; i < len; i++) {
						const rendered = await cs.renderDisplay(region.iterator, entriesJson, i);
						cards.push(...rendered);
					}
					newRegionData.set(region.id, { id: region.id, name: region.name, cards });
				}
			} catch (e) {
				console.warn('[renderRegions] region failed:', region.id, e);
			}
		}
		st.regionData = newRegionData;
	}

	// --- Infinite scroll (upward) ---

	let observer: IntersectionObserver | null = null;

	function setupObserver() {
		if (observer) observer.disconnect();
		if (!sentinelEl) return;
		observer = new IntersectionObserver(
			(entries) => {
				if (entries[0]?.isIntersecting && !loadingMore && st.loadedFrom > 0) {
					loadMore();
				}
			},
			{ threshold: 0.1 }
		);
		observer.observe(sentinelEl);
	}

	$effect(() => {
		if (sentinelEl && useDisplayEngine) {
			setupObserver();
		}
		return () => {
			if (observer) {
				observer.disconnect();
				observer = null;
			}
		};
	});

	async function loadMore() {
		if (!st.chatSession || st.chatSession.freed || st.chatSession.busy || loadingMore || st.loadedFrom <= 0 || st.isLoading) return;
		loadingMore = true;

		const newStart = Math.max(0, st.loadedFrom - 5);
		const newEnd = st.loadedFrom;

		try {
			const prevHeight = viewportEl?.scrollHeight ?? 0;

			const olderCards = await renderDisplayRange(st.chatSession, newStart, newEnd);
			st.displayCards = [...olderCards, ...st.displayCards];
			st.loadedFrom = newStart;

			// Preserve scroll position
			await tick();
			if (viewportEl) {
				const newHeight = viewportEl.scrollHeight;
				viewportEl.scrollTop += newHeight - prevHeight;
			}
		} finally {
			loadingMore = false;
		}
	}

	// --- Submit / Cancel ---

	async function cancelTurn() {
		const ok = await confirmAction('Cancel the current turn?', 'Cancel Turn');
		if (!ok) return;
		st.cancelled = true;
		// Bump turn ID so the old turn's finally won't clobber our state.
		st.activeTurnId++;
		// Immediately release lock and reset UI — don't wait for WASM to finish.
		uiState.unlock(st.turnDeps);
		st.isLoading = false;
		// Destroy session.
		st.chatSession = null;
		st.chatSessionKey = null;
		// Unblock any pending resolver so the WASM turn can finish in background.
		if (st.pendingResolve) {
			st.pendingResolve.resolve('');
			st.pendingResolve = null;
			st.pendingValue = '';
		}
	}

	async function submit() {
		if (st.isLoading) {
			cancelTurn();
			return;
		}
		if (!canSubmit) return;
		const currentInput = inputValue;
		const currentInputParam = inputParam;

		st.isLoading = true;
		st.cancelled = false;
		st.runtimeError = '';
		let turnSession: ChatSession | null = null;
		const myTurnId = ++st.activeTurnId;

		try {
			const cs = await ensureChatSession();
			const snapshotKey = st.chatSessionKey;
			turnSession = cs;

			const resolver = (key: string): string | Promise<string> => {
				if (key === currentInputParam) return currentInput;
				return new Promise<string>((resolve) => {
					st.pendingResolve = { key, resolve };
					st.pendingValue = '';
				});
			};

			st.lastInput = { param: currentInputParam!, value: currentInput };
			st.turnDeps = deps;
			uiState.lock(st.turnDeps);

			// Start streaming evaluation
			await cs.startEvaluate('main', false, resolver);

			if (st.chatSessionKey !== snapshotKey) {
				cs.cancelEvaluate();
				return;
			}

			inputValue = '';

			// Stream results one by one
			while (true) {
				if (myTurnId !== st.activeTurnId) {
					cs.cancelEvaluate();
					return;
				}

				const item = await cs.evaluateNext(resolver);
				if (item === null) break;

				// Each item is a concrete value — extract display card
				const card = item as Record<string, unknown>;
				if (typeof card?.v === 'string') {
					st.displayCards = [...st.displayCards, { name: '', content: card.v }];
				}
				scrollToBottom();
			}

			// Evaluation complete — update tree state
			const treeView = cs.tree();
			if (treeView) {
				st.treeNodes = treeView.nodes;
				st.treeCursor = treeView.cursor;
			}
			st.turnCount = await cs.turnCount();

			await renderRegions(cs);
		} catch (err) {
			// Stale turn (cancelled or superseded) — just clean up WASM.
			if (myTurnId !== st.activeTurnId) {
				turnSession?.free();
				return;
			}
			st.runtimeError = err instanceof Error ? err.message : String(err);
			// WASM crash: session is unusable — discard so next submit recreates it.
			if (turnSession?.crashed) {
				st.chatSession = null;
				st.chatSessionKey = null;
			}
		} finally {
			turnSession?.finishEvaluate();
			// Only touch shared state if this is still the active turn.
			if (myTurnId === st.activeTurnId) {
				uiState.unlock(st.turnDeps);
				st.isLoading = false;
				st.cancelled = false;
				st.pendingResolve = null;
				st.pendingValue = '';
			}
			scrollToBottom();
		}
	}

	function submitPending() {
		if (!st.pendingResolve || !st.pendingValue.trim()) return;
		st.pendingResolve.resolve(st.pendingValue.trim());
		st.pendingResolve = null;
		st.pendingValue = '';
	}

	async function handleHistoryUndo() {
		if (!st.chatSession || st.isLoading) return;
		await st.chatSession.undo();
		await refreshAfterNavigation(st.chatSession);
	}

	async function handleReroll() {
		if (!canReroll || !st.lastInput) return;
		const ok = await confirmAction('Undo the last turn and retry with the same input?', 'Reroll');
		if (!ok) return;

		await handleHistoryUndo();
		inputValue = st.lastInput.value;
		await submit();
	}

	async function handleHistoryGoto(id: string) {
		if (!st.chatSession || st.isLoading) return;
		await st.chatSession.goto(id);
		await refreshAfterNavigation(st.chatSession);
	}

	/** Refresh display + tree after cursor navigation (undo/goto). */
	async function refreshAfterNavigation(cs: ChatSession) {
		// Update tree cursor from WASM
		const treeView = cs.tree();
		if (treeView) {
			st.treeNodes = treeView.nodes;
			st.treeCursor = treeView.cursor;
		}
		st.turnCount = await cs.turnCount();
		if (useDisplayEngine) {
			await initialDisplayLoad(cs);
		} else {
			st.displayCards = [];
		}
		await renderRegions(cs);
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			if (st.pendingResolve) {
				submitPending();
			} else {
				submit(); // handles both submit and cancel (when isLoading)
			}
		}
	}
</script>

{#snippet historyPanel()}
	<ScrollArea class="h-full" bind:viewportRef={viewportEl}>
		<div
			class="flex flex-col gap-3 p-4 md:p-6 lg:px-8 max-w-4xl mx-auto w-full"
		>
			{#if useDisplayEngine && st.loadedFrom > 0}
				<div bind:this={sentinelEl} class="flex justify-center py-2">
					{#if loadingMore}
						<Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
					{:else}
						<span class="text-xs text-muted-foreground">Scroll up for more</span>
					{/if}
				</div>
			{/if}
			{#if st.displayCards.length === 0}
				<div
					class="flex flex-col items-center justify-center gap-4 py-20 text-muted-foreground animate-in fade-in slide-in-from-bottom-4 duration-500"
				>
					<div class="rounded-full bg-secondary/50 p-4">
						<MessageSquarePlus class="h-8 w-8 text-primary" />
					</div>
					<div class="flex flex-col items-center gap-1 text-center">
						<div class="text-xl font-medium text-foreground">Start a conversation</div>
						<div class="text-sm">Configure your blocks and type a message below.</div>
					</div>
				</div>
			{:else}
				{#each st.displayCards as card, i (i)}
					<DisplayCard {card} />
				{/each}
			{/if}
			<div></div>
		</div>
	</ScrollArea>
{/snippet}

{#snippet regionPanel(region: RenderedRegion)}
	<ScrollArea class="h-full">
		<div class="flex flex-col gap-2 p-3">
			<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground px-1">{region.name}</span>
			{#each region.cards as card, i (i)}
				<DisplayCard {card} compact />
			{/each}
		</div>
	</ScrollArea>
{/snippet}

<BasePage {deps} onConfigChange={handleConfigChange} debounceMs={800} lockable={false}>
{#if !isConfigured}
	<div class="flex h-full items-center justify-center text-sm text-muted-foreground">
		{#if !hasHistoryBinding}
			Set up a history binding in the prompt to enable chat.
		{:else if !inputParam}
			Select an input parameter in the prompt to enable chat.
		{/if}
	</div>
{:else}
{#if bot.embeddedStyle}
	{@html `<style>${bot.embeddedStyle}</style>`}
{/if}
<div class="flex h-full">
	<!-- Main chat area -->
	<div class="flex flex-1 flex-col min-w-0">
	<div class="flex-1 overflow-hidden flex items-center justify-center">
		{#if hasRegions}
			<div
				class="chat-grid"
				class:chat-grid-aspect={hasAspect}
				style="{computedGridStyle} {hasAspect ? `aspect-ratio: ${layout.aspect};` : ''}"
			>
				{#each placements as p (p.id)}
					<div
						class="grid-area overflow-hidden"
						class:border-r={p.colEnd <= layout.colSizes.length}
						class:border-b={p.rowEnd <= layout.rowSizes.length}
						style="grid-column: {p.colStart} / {p.colEnd}; grid-row: {p.rowStart} / {p.rowEnd};"
					>
						{#if p.id === GRID_HISTORY}
							{@render historyPanel()}
						{:else}
							{@const region = st.regionData.get(p.id)}
							{#if region}
								{@render regionPanel(region)}
							{:else}
								<div class="flex h-full items-center justify-center text-xs text-muted-foreground">
									Empty region
								</div>
							{/if}
						{/if}
					</div>
				{/each}
			</div>
		{:else}
			{@render historyPanel()}
		{/if}
	</div>

	{#if errorMsg}
		<div class="shrink-0 px-4 pt-2">
			<div class="mx-auto max-w-4xl rounded-lg border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive whitespace-pre-wrap">
				{errorMsg}
			</div>
		</div>
	{/if}
	<div class="shrink-0 p-4 border-t">
		<div class="mx-auto flex max-w-4xl gap-2 rounded-xl border bg-background p-2 focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 transition-all shadow-sm">
			<div class="flex flex-1 flex-col gap-2">
				{#if st.pendingResolve}
					<Label class="text-xs text-muted-foreground px-1 pl-2 pt-1 font-medium">@{st.pendingResolve.key}</Label>
					<Textarea
						value={st.pendingValue}
						oninput={(e) => { st.pendingValue = e.currentTarget.value; }}
						onkeydown={handleKeydown}
						placeholder={`@${st.pendingResolve.key}... (Enter to send)`}
						rows={2}
						class="min-h-[44px] resize-none border-0 focus-visible:ring-0 shadow-none bg-transparent pt-3 pb-3"
					/>
				{:else}
					<Textarea
						value={inputValue}
						oninput={(e) => { inputValue = e.currentTarget.value; }}
						onkeydown={handleKeydown}
						placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
						rows={2}
						readonly={st.isLoading}
						class="min-h-[44px] resize-none border-0 focus-visible:ring-0 shadow-none bg-transparent pt-3 pb-3 {st.isLoading ? 'opacity-50' : ''}"
					/>
				{/if}
			</div>
			<div class="flex flex-row items-end gap-2 pb-1 pr-1">
				{#if st.pendingResolve}
					<Button size="icon" class="h-10 w-10 shrink-0 rounded-lg transition-transform hover:scale-105 active:scale-95" disabled={!st.pendingValue.trim()} onclick={submitPending}>
						<Send class="h-4 w-4" />
					</Button>
				{:else if st.isLoading}
					<Button size="icon" variant="destructive" class="h-10 w-10 shrink-0 rounded-lg transition-transform hover:scale-105 active:scale-95" onclick={submit}>
						<Square class="h-4 w-4" />
					</Button>
				{:else}
					{#if canReroll}
						<Button size="icon" variant="ghost" class="h-10 w-10 shrink-0 rounded-lg transition-transform hover:scale-105 active:scale-95 text-muted-foreground hover:text-foreground" onclick={handleReroll}>
							<RotateCcw class="h-4 w-4" />
						</Button>
					{/if}
					<Button size="icon" class="h-10 w-10 shrink-0 rounded-lg transition-transform hover:scale-105 active:scale-95" disabled={!canSubmit} onclick={submit}>
						<Send class="h-4 w-4" />
					</Button>
				{/if}
			</div>
		</div>
	</div>
	</div>

</div>
{/if}
</BasePage>

<style>
	.chat-grid {
		display: grid;
		width: 100%;
		height: 100%;
	}
	.chat-grid-aspect {
		width: auto;
		height: auto;
		max-width: 100%;
		max-height: 100%;
	}
	.grid-area {
		min-width: 0;
		min-height: 0;
	}
</style>
