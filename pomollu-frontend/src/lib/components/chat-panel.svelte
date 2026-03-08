<script lang="ts">
	import type { RenderedRegion, RenderedCard, GridLayout, Session, Bot } from '$lib/types.js';
	import { GRID_HISTORY, HISTORY_BINDING_NAME, gridStyle, computePlacements } from '$lib/types.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Button } from '$lib/components/ui/button';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Label } from '$lib/components/ui/label';
	import { Send, MessageSquarePlus, Loader2 } from 'lucide-svelte';
	import { tick, onMount, onDestroy } from 'svelte';
	import DisplayCard from './display-card.svelte';
	import { sessionStore, promptStore, uiState } from '$lib/stores.svelte.js';
	import { ChatSession } from '$lib/engine.js';
	import { buildSessionConfig } from '$lib/session-builder.js';

	let {
		session,
		bot,
	}: {
		session: Session;
		bot: Bot;
	} = $props();

	let prompt = $derived(promptStore.get(bot.promptId));
	let inputParam = $derived(prompt?.inputParam);
	let hasHistoryBinding = $derived(
		prompt?.contextBindings.some((b) => b.name === HISTORY_BINDING_NAME && b.script.trim() !== '') ?? false
	);
	let isConfigured = $derived(hasHistoryBinding && !!inputParam);
	let inputValue = $state('');
	let canSubmit = $derived(inputValue.trim().length > 0 && isConfigured);
	let isLoading = $state(false);
	let errorMsg = $state('');

	// On-demand resolver: engine requests a key → show input → wait for user
	let pendingResolve: { key: string; resolve: (value: string) => void } | null = $state(null);
	let pendingValue = $state('');

	// Engine result: entrypoint output after each turn
	let turnCount = $state(0);

	// Display cards
	let displayCards = $state<RenderedCard[]>([]);
	let totalListLen = $state(0);
	let loadedFrom = $state(0);
	let loadingMore = $state(false);
	let sentinelEl = $state<HTMLElement>();
	let scrollContainerEl = $state<HTMLElement>();

	// Whether display engine is active
	let useDisplayEngine = $derived(
		bot.display.iterator.trim() !== '' && bot.display.entries.length > 0
	);

	// WASM ChatSession — lazy-initialized per session
	let chatSession: ChatSession | null = null;
	let chatSessionKey: string | null = null;
	let destroyed = false;

	async function ensureChatSession(): Promise<ChatSession> {
		const key = `${bot.id}:${session.id}`;
		if (chatSession && chatSessionKey === key) return chatSession;
		if (chatSession) {
			chatSession.free();
			chatSession = null;
		}

		const result = buildSessionConfig(bot);
		if (!result) throw new Error('no nodes configured');
		if (!result.ok) throw new Error(result.errors.join('\n'));

		// No callback — storage is synced explicitly after create() and turn().
		chatSession = await ChatSession.create(result.config, session.storage);
		chatSessionKey = key;

		// Sync storage (ChatEngine::new sets "context" metadata).
		sessionStore.update(session.id, (s) => ({ ...s, storage: chatSession!.exportStorageJson() }));
		turnCount = chatSession.turnCount();

		if (useDisplayEngine) {
			await initialDisplayLoad(chatSession);
		}

		return chatSession;
	}

	async function initialDisplayLoad(cs: ChatSession) {
		try {
			const len = await cs.displayListLen(bot.display.iterator);
			totalListLen = len;
			loadedFrom = Math.max(0, len - 10);
			if (len > 0) {
				displayCards = await renderDisplayRange(cs, loadedFrom, len);
			}
		} catch {
			// display engine failed — fall through to empty
		}
	}

	onMount(async () => {
		if (!useDisplayEngine && !hasRegions) return;
		try {
			const cs = await ensureChatSession();
			if (destroyed) return;
			if (useDisplayEngine) {
				await initialDisplayLoad(cs);
				if (destroyed) return;
			}
			await renderRegions(cs);
		} catch {
			// no config or session yet
		}
	});

	onDestroy(() => {
		destroyed = true;
		if (chatSession) {
			chatSession.free(); // safe — defers if turn() is in progress
			chatSession = null;
		}
	});

	// Re-render static regions when bot.regions changes
	let regionsKey = $derived(JSON.stringify(bot.regions));
	$effect(() => {
		void regionsKey;
		if (!isConfigured || destroyed || isLoading) return;
		ensureChatSession().then((cs) => {
			if (!destroyed && !isLoading) renderRegions(cs);
		}).catch((e) => {
			console.warn('[regions effect] ensureChatSession failed:', e);
		});
	});

	// Layout from bot
	let layout = $derived<GridLayout>(bot.layout);
	let placements = $derived(computePlacements(layout));
	let hasRegions = $derived(placements.some((p) => p.id !== GRID_HISTORY));
	let hasAspect = $derived((layout.aspect ?? 0) > 0);
	let computedGridStyle = $derived(gridStyle(layout));

	let regionData = $state<Map<string, RenderedRegion>>(new Map());

	let bottomEl = $state<HTMLElement>();

	async function scrollToBottom() {
		await tick();
		if (bottomEl) {
			bottomEl.scrollIntoView({ behavior: 'smooth' });
		}
	}

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
		const newRegionData = new Map(regionData);
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
		regionData = newRegionData;
	}

	// --- Infinite scroll (upward) ---

	let observer: IntersectionObserver | null = null;

	function setupObserver() {
		if (observer) observer.disconnect();
		if (!sentinelEl) return;
		observer = new IntersectionObserver(
			(entries) => {
				if (entries[0]?.isIntersecting && !loadingMore && loadedFrom > 0) {
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
		if (!chatSession || loadingMore || loadedFrom <= 0 || isLoading) return;
		loadingMore = true;

		const newStart = Math.max(0, loadedFrom - 10);
		const newEnd = loadedFrom;

		try {
			const scrollEl = scrollContainerEl;
			const prevHeight = scrollEl?.scrollHeight ?? 0;

			const olderCards = await renderDisplayRange(chatSession, newStart, newEnd);
			displayCards = [...olderCards, ...displayCards];
			loadedFrom = newStart;

			// Preserve scroll position
			await tick();
			if (scrollEl) {
				const newHeight = scrollEl.scrollHeight;
				scrollEl.scrollTop += newHeight - prevHeight;
			}
		} finally {
			loadingMore = false;
		}
	}

	// --- Submit ---

	async function submit() {
		if (!canSubmit || isLoading) return;
		const currentInput = inputValue;
		const currentInputParam = inputParam;
		inputValue = '';

		isLoading = true;
		errorMsg = '';

		try {
			const cs = await ensureChatSession();

			// Static params are handled as Expr nodes by the engine.
			// Resolver only handles dynamic params.
			const resolver = (key: string): string | Promise<string> => {
				if (key === currentInputParam) return currentInput;
				// On-demand: show input field and wait for user
				return new Promise<string>((resolve) => {
					pendingResolve = { key, resolve };
					pendingValue = '';
				});
			};

			uiState.busyBotId = bot.id;
			let result: unknown;
			try {
				result = await cs.turn(resolver);
			} finally {
				uiState.busyBotId = null;
			}
			if (destroyed) return;

			// Sync full storage to Svelte after turn completes.
			sessionStore.update(session.id, (s) => ({ ...s, storage: cs.exportStorageJson() }));

			turnCount = cs.turnCount();

			if (useDisplayEngine) {
				// Render new display entries
				const newLen = await cs.displayListLen(bot.display.iterator);
				if (newLen > totalListLen) {
					const newCards = await renderDisplayRange(cs, totalListLen, newLen);
					displayCards = [...displayCards, ...newCards];
					totalListLen = newLen;
				}
			} else {
				// Fallback: show input + result as cards
				const content = formatResult(result);
				displayCards = [...displayCards, { name: 'User', content: currentInput }, { name: 'Assistant', content }];
			}

			// Render static regions
			await renderRegions(cs);

		} catch (err) {
			errorMsg = err instanceof Error ? err.message : String(err);
		} finally {
			isLoading = false;
			scrollToBottom();
		}
	}

	function submitPending() {
		if (!pendingResolve || !pendingValue.trim()) return;
		pendingResolve.resolve(pendingValue.trim());
		pendingResolve = null;
		pendingValue = '';
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			if (pendingResolve) {
				submitPending();
			} else {
				submit();
			}
		}
	}
</script>

{#snippet historyPanel()}
	<ScrollArea class="h-full">
		<div
			bind:this={scrollContainerEl}
			class="flex flex-col gap-3 p-4 md:p-6 lg:px-8 max-w-4xl mx-auto w-full"
		>
			{#if useDisplayEngine && loadedFrom > 0}
				<div bind:this={sentinelEl} class="flex justify-center py-2">
					{#if loadingMore}
						<Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
					{:else}
						<span class="text-xs text-muted-foreground">Scroll up for more</span>
					{/if}
				</div>
			{/if}
			{#if displayCards.length === 0}
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
				{#each displayCards as card, i (i)}
					<DisplayCard {card} />
				{/each}
			{/if}
			<div bind:this={bottomEl}></div>
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
<div class="flex h-full flex-col">
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
							{@const region = regionData.get(p.id)}
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
			<div class="mx-auto max-w-4xl rounded-lg border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
				{errorMsg}
			</div>
		</div>
	{/if}
	<div class="shrink-0 p-4 border-t">
		<div class="mx-auto flex max-w-4xl gap-2 rounded-xl border bg-background p-2 focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 transition-all shadow-sm">
			<div class="flex flex-1 flex-col gap-2">
				{#if pendingResolve}
					<Label class="text-xs text-muted-foreground px-1 pl-2 pt-1 font-medium">@{pendingResolve.key}</Label>
					<Textarea
						value={pendingValue}
						oninput={(e) => { pendingValue = e.currentTarget.value; }}
						onkeydown={handleKeydown}
						placeholder={`@${pendingResolve.key}... (Enter to send)`}
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
						class="min-h-[44px] resize-none border-0 focus-visible:ring-0 shadow-none bg-transparent pt-3 pb-3"
					/>
				{/if}
			</div>
			<div class="flex flex-col justify-end pb-1 pr-1">
				{#if pendingResolve}
					<Button size="icon" class="h-10 w-10 shrink-0 rounded-lg transition-transform hover:scale-105 active:scale-95" disabled={!pendingValue.trim()} onclick={submitPending}>
						<Send class="h-4 w-4" />
					</Button>
				{:else}
					<Button size="icon" class="h-10 w-10 shrink-0 rounded-lg transition-transform hover:scale-105 active:scale-95" disabled={!canSubmit || isLoading} onclick={submit}>
						{#if isLoading}
							<Loader2 class="h-4 w-4 animate-spin" />
						{:else}
							<Send class="h-4 w-4" />
						{/if}
					</Button>
				{/if}
			</div>
		</div>
	</div>
</div>
{/if}

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
