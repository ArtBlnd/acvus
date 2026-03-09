<script lang="ts">
	import type { DisplayEntry, DisplayRegion, GridLayout, BotDisplay, ContextParam } from '$lib/types.js';
	import { GRID_HISTORY, HISTORY_ENTRY_TYPE, CONTEXT_TYPE, createDefaultLayout } from '$lib/types.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Input } from '$lib/components/ui/input';
	import { Textarea } from '$lib/components/ui/textarea';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import { Button } from '$lib/components/ui/button';
	import { Separator } from '$lib/components/ui/separator';
	import { Plus, X, Download } from 'lucide-svelte';
	import { downloadJson } from '$lib/io.js';
	import { botStore, promptStore, profileStore, providerStore, uiState, createId } from '$lib/stores.svelte.js';
	import AcvusEngineField from './acvus-engine-field.svelte';
	import GridLayoutEditor from './grid-layout-editor.svelte';
	import ContextParamsEditor from './context-params-editor.svelte';
	import { analyzeBot } from '$lib/param-resolver.js';
	import { analyzeWithTypes } from '$lib/engine.js';
	import { onDestroy } from 'svelte';

	let { botId }: { botId: string } = $props();

	let bot = $derived(botStore.get(botId));
	let prompts = $derived(promptStore.prompts);
	let profiles = $derived(profileStore.profiles);
	let hasOrphanPrompt = $derived(bot != null && bot.promptId !== '' && !promptStore.get(bot.promptId));
	let hasOrphanProfile = $derived(bot != null && bot.profileId !== '' && !profileStore.get(bot.profileId));

	// --- Display entry CRUD ---

	function updateDisplay<K extends keyof BotDisplay>(field: K, value: BotDisplay[K]) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({ ...b, display: { ...b.display, [field]: value } }));
	}

	function addEntry() {
		if (!bot) return;
		const entry: DisplayEntry = { id: createId(), name: '', condition: '', template: '' };
		botStore.update(bot.id, (b) => ({ ...b, display: { ...b.display, entries: [...b.display.entries, entry] } }));
	}

	function updateEntry(entryId: string, patch: Partial<DisplayEntry>) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({
			...b,
			display: { ...b.display, entries: b.display.entries.map((e) => e.id === entryId ? { ...e, ...patch } : e) }
		}));
	}

	function removeEntry(entryId: string) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({
			...b,
			display: { ...b.display, entries: b.display.entries.filter((e) => e.id !== entryId) }
		}));
	}

	// --- Region management ---

	function addRegion(kind: 'iterable' | 'static') {
		if (!bot) return;
		const base = { id: createId(), name: '' };
		const region: DisplayRegion = kind === 'iterable'
			? { ...base, kind: 'iterable', iterator: '', entries: [] }
			: { ...base, kind: 'static', template: '' };
		botStore.update(bot.id, (b) => ({ ...b, regions: [...b.regions, region] }));
	}

	function updateRegion(regionId: string, patch: Record<string, unknown>) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({
			...b,
			regions: b.regions.map((r) => r.id === regionId ? { ...r, ...patch } as DisplayRegion : r)
		}));
	}

	function switchRegionKind(regionId: string, kind: 'iterable' | 'static') {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({
			...b,
			regions: b.regions.map((r) => {
				if (r.id !== regionId) return r;
				if (kind === 'iterable') {
					return { id: r.id, name: r.name, kind: 'iterable', iterator: '', entries: [] };
				}
				return { id: r.id, name: r.name, kind: 'static', template: '' };
			})
		}));
	}

	function removeRegion(regionId: string) {
		if (!bot) return;
		botStore.update(bot.id, (b) => {
			const layout = b.layout;
			return {
				...b,
				regions: b.regions.filter((r) => r.id !== regionId),
				layout: {
					...layout,
					areas: layout.areas.map((row) => row.map((cell) => cell === regionId ? '' : cell))
				}
			};
		});
	}

	function addRegionEntry(regionId: string) {
		if (!bot) return;
		const entry: DisplayEntry = { id: createId(), name: '', condition: '', template: '' };
		botStore.update(bot.id, (b) => ({
			...b,
			regions: b.regions.map((r) =>
				r.id === regionId && r.kind === 'iterable' ? { ...r, entries: [...r.entries, entry] } : r
			)
		}));
	}

	function updateRegionEntry(regionId: string, entryId: string, patch: Partial<DisplayEntry>) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({
			...b,
			regions: b.regions.map((r) =>
				r.id === regionId && r.kind === 'iterable' ? { ...r, entries: r.entries.map((e) => e.id === entryId ? { ...e, ...patch } : e) } : r
			)
		}));
	}

	function removeRegionEntry(regionId: string, entryId: string) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({
			...b,
			regions: b.regions.map((r) =>
				r.id === regionId && r.kind === 'iterable' ? { ...r, entries: r.entries.filter((e) => e.id !== entryId) } : r
			)
		}));
	}

	function updateLayout(newLayout: GridLayout) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({ ...b, layout: newLayout }));
	}

	// --- Unresolved params analysis (full hierarchy) ---

	let analyzeTimer: ReturnType<typeof setTimeout> | null = null;
	let discoveredContextTypes = $state<Record<string, import('$lib/type-parser.js').TypeDesc>>({});
	let analysisErrors = $state<string[]>([]);

	function runAnalysis() {
		if (!bot) throw new Error('bot not found');
		const prompt = promptStore.get(bot.promptId);
		if (!prompt) throw new Error(`prompt '${bot.promptId}' not found`);
		const profile = profileStore.get(bot.profileId);
		if (!profile) throw new Error(`profile '${bot.profileId}' not found`);

		const result = analyzeBot(bot, prompt, profile, (id) => {
			const p = providerStore.get(id);
			if (!p) throw new Error(`provider '${id}' not found`);
			return p.api;
		});
		if (!result.ok) {
			analysisErrors = result.errors;
			discoveredContextTypes = {};
			return;
		}
		analysisErrors = [];
		discoveredContextTypes = result.env.contextTypes;
		botStore.update(bot.id, (b) => ({
			...b, contextParams: result.params
		}));
	}

	function scheduleAnalysis() {
		if (analyzeTimer) clearTimeout(analyzeTimer);
		analyzeTimer = setTimeout(() => {
			analyzeTimer = null;
			runAnalysis();
		}, 200);
	}

	// Stable key: re-analyze when content or user-set param values change.
	// Includes resolution/userType (not inferredType/active) to trigger liveness re-check
	// when static param values change, without causing circular dependency.
	const paramUserKey = (params?: ContextParam[]) =>
		params?.map((p) => [p.name, p.resolution, p.userType]);

	let analysisKey = $derived(JSON.stringify([
		bot?.display,
		bot?.regions,
		bot?.children,
		bot?.promptId,
		bot?.profileId,
		promptStore.get(bot?.promptId ?? '')?.contextBindings,
		promptStore.get(bot?.promptId ?? '')?.children,
		profileStore.get(bot?.profileId ?? '')?.children,
		paramUserKey(bot?.contextParams),
		paramUserKey(promptStore.get(bot?.promptId ?? '')?.contextParams),
		paramUserKey(profileStore.get(bot?.profileId ?? '')?.contextParams),
	]));

	let isFirstRun = true;

	$effect(() => {
		void analysisKey;
		if (isFirstRun) {
			isFirstRun = false;
			runAnalysis();
		} else {
			scheduleAnalysis();
		}
	});

	onDestroy(() => {
		if (analyzeTimer) clearTimeout(analyzeTimer);
	});

	function handleParamsUpdate(params: ContextParam[]) {
		if (!bot) return;
		botStore.update(bot.id, (b) => ({ ...b, contextParams: params }));
	}

	function handleTypeChange(_name: string, _type: string) {
		scheduleAnalysis();
	}

	let displayContextTypes = $derived.by(() => {
		const result = { ...discoveredContextTypes };
		for (const p of bot?.contextParams ?? []) {
			if (p.resolution.kind === 'dynamic') {
				delete result[p.name];
			}
		}
		// @history is always available from prompt binding with fixed type
		result['history'] = HISTORY_ENTRY_TYPE;
		return result;
	});

	function computeIterableEntryTypes(
		iterator: string,
		baseTypes: Record<string, import('$lib/type-parser.js').TypeDesc>
	): Record<string, import('$lib/type-parser.js').TypeDesc> {
		if (!iterator.trim()) return baseTypes;
		const result = analyzeWithTypes(iterator, 'script', baseTypes);
		if (!result.ok || !result.tail_type) return baseTypes;
		if (result.tail_type.kind !== 'list') return baseTypes;
		return { ...baseTypes, item: result.tail_type.elem, index: { kind: 'primitive', name: 'Int' } };
	}

	let locked = $derived(uiState.isBotBusy(botId));
</script>

<div class="flex h-full flex-col" class:pointer-events-none={locked} class:opacity-60={locked}>
	{#if locked}
		<div class="shrink-0 border-b bg-amber-500/10 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">Turn in progress — editing locked</div>
	{/if}
	<div class="flex items-center justify-between shrink-0 border-b px-4 py-2">
		<span class="text-sm font-medium">Bot Settings</span>
		{#if bot}
			<div class="flex items-center gap-1">
				<Button variant="ghost" size="icon-sm" class="text-muted-foreground" onclick={() => downloadJson(bot, `${bot.name}.bot.json`)} title="Export">
					<Download class="h-3.5 w-3.5" />
				</Button>
				<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={() => uiState.removeBot(bot.id)} title="Delete bot">
					&times;
				</Button>
			</div>
		{/if}
	</div>

	{#if bot}
		<ScrollArea class="flex-1">
			<div class="mx-auto max-w-2xl space-y-4 px-6 py-10">
				<div class="space-y-1">
					<Label>Name</Label>
					<Input
						value={bot.name}
						oninput={(e) => botStore.update(bot.id, (b) => ({ ...b, name: e.currentTarget.value }))}
					/>
				</div>

				<div class="space-y-1">
					<Label>Prompt</Label>
					<Select.Root
						type="single"
						value={bot.promptId}
						onValueChange={(v) => { if (v) botStore.update(bot.id, (b) => ({ ...b, promptId: v })); }}
					>
						<Select.Trigger class="w-full {hasOrphanPrompt ? 'border-destructive' : ''}">
							{promptStore.get(bot.promptId)?.name ?? 'Select prompt...'}
						</Select.Trigger>
						<Select.Content>
							{#each prompts as p}
								<Select.Item value={p.id}>{p.name}</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
					{#if hasOrphanPrompt}
						<p class="text-xs text-destructive">Prompt has been deleted. Please select another.</p>
					{/if}
				</div>

				<div class="space-y-1">
					<Label>Profile</Label>
					<Select.Root
						type="single"
						value={bot.profileId}
						onValueChange={(v) => { if (v) botStore.update(bot.id, (b) => ({ ...b, profileId: v })); }}
					>
						<Select.Trigger class="w-full {hasOrphanProfile ? 'border-destructive' : ''}">
							{profileStore.get(bot.profileId)?.name ?? 'Select profile...'}
						</Select.Trigger>
						<Select.Content>
							{#each profiles as p}
								<Select.Item value={p.id}>{p.name}</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
					{#if hasOrphanProfile}
						<p class="text-xs text-destructive">Profile has been deleted. Please select another.</p>
					{/if}
				</div>

				<Separator />

				<div class="space-y-3">
					<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Embedded Style</span>
					<p class="text-xs text-muted-foreground">CSS styles applied to the chat display. Use this to style HTML elements in templates.</p>
					<Textarea
						value={bot.embeddedStyle ?? ''}
						oninput={(e) => botStore.update(bot.id, (b) => ({ ...b, embeddedStyle: e.currentTarget.value }))}
						placeholder={".card-body h2 { color: var(--color-primary); }"}
						rows={4}
						class="font-mono text-xs"
					/>
				</div>

				<Separator />

				<div class="space-y-3">
					<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Layout</span>
					<p class="text-xs text-muted-foreground">Configure how regions are arranged in the chat view.</p>
					<GridLayoutEditor layout={bot.layout} regions={bot.regions} onupdate={updateLayout} />
				</div>

				<Separator />

				<div class="space-y-3">
					<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">History Display</span>

					<div class="space-y-1">
						<Label>Iterator</Label>
						<AcvusEngineField
							mode="script"
							placeholder="e.g. @messages"
							value={bot.display.iterator}
							oninput={(v) => updateDisplay('iterator', v)}
							contextTypes={displayContextTypes}
							{analysisErrors}
						/>
						<p class="text-xs text-muted-foreground">Expression that produces a list to iterate over.</p>
					</div>

					<div class="space-y-2">
						<Label>Entries</Label>
						{#each bot.display.entries as entry (entry.id)}
							<div class="rounded-md border p-3 space-y-2">
								<div class="flex items-center gap-2">
									<Input
										class="flex-1 text-sm"
										value={entry.name}
										oninput={(e) => updateEntry(entry.id, { name: e.currentTarget.value })}
										placeholder="Display name..."
									/>
									<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive shrink-0" onclick={() => removeEntry(entry.id)}>
										<X class="h-3.5 w-3.5" />
									</Button>
								</div>
								<div class="space-y-1">
									<Label class="text-xs">Condition</Label>
									<AcvusEngineField
										mode="script"
										placeholder="e.g. @item.role == 'user'"
										value={entry.condition}
										oninput={(v) => updateEntry(entry.id, { condition: v })}
										contextTypes={computeIterableEntryTypes(bot.display.iterator, displayContextTypes)}
										{analysisErrors}
									/>
								</div>
								<div class="space-y-1">
									<Label class="text-xs">Template</Label>
									<AcvusEngineField
										mode="template"
										placeholder="Template content..."
										value={entry.template}
										oninput={(v) => updateEntry(entry.id, { template: v })}
										contextTypes={computeIterableEntryTypes(bot.display.iterator, displayContextTypes)}
										{analysisErrors}
									/>
								</div>
							</div>
						{/each}
						<Button variant="outline" size="sm" class="w-full border-dashed text-muted-foreground" onclick={addEntry}>
							<Plus class="h-3 w-3 mr-1" /> Add Entry
						</Button>
					</div>
				</div>

				<Separator />

				<div class="space-y-3">
					<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Regions</span>
					<p class="text-xs text-muted-foreground">Define display regions, then place them on the layout grid above.</p>

					{#each bot.regions as region (region.id)}
						<div class="rounded-lg border p-3 space-y-3">
							<div class="flex items-center gap-2">
								<Input
									class="flex-1 text-sm"
									value={region.name}
									oninput={(e) => updateRegion(region.id, { name: e.currentTarget.value })}
									placeholder="Region name..."
								/>
								<Select.Root
									type="single"
									value={region.kind}
									onValueChange={(v) => { if (v) switchRegionKind(region.id, v as 'iterable' | 'static'); }}
								>
									<Select.Trigger class="w-28 text-xs">
										{region.kind === 'iterable' ? 'Iterable' : 'Static'}
									</Select.Trigger>
									<Select.Content>
										<Select.Item value="iterable">Iterable</Select.Item>
										<Select.Item value="static">Static</Select.Item>
									</Select.Content>
								</Select.Root>
								<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive shrink-0" onclick={() => removeRegion(region.id)}>
									<X class="h-3.5 w-3.5" />
								</Button>
							</div>

							{#if region.kind === 'static'}
								<div class="space-y-1">
									<Label class="text-xs">Template</Label>
									<AcvusEngineField
										mode="template"
										placeholder="Template content..."
										value={region.template}
										oninput={(v) => updateRegion(region.id, { template: v })}
										contextTypes={displayContextTypes}
										{analysisErrors}
									/>
								</div>
							{:else}
								<div class="space-y-1">
									<Label class="text-xs">Iterator</Label>
									<AcvusEngineField
										mode="script"
										placeholder="e.g. @status"
										value={region.iterator}
										oninput={(v) => updateRegion(region.id, { iterator: v })}
										contextTypes={displayContextTypes}
										{analysisErrors}
									/>
								</div>

								<div class="space-y-2">
									<Label class="text-xs">Entries</Label>
									{#each region.entries as entry (entry.id)}
										<div class="rounded-md border p-2 space-y-2 bg-muted/30">
											<div class="flex items-center gap-2">
												<Input
													class="flex-1 text-xs"
													value={entry.name}
													oninput={(e) => updateRegionEntry(region.id, entry.id, { name: e.currentTarget.value })}
													placeholder="Display name..."
												/>
												<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive shrink-0" onclick={() => removeRegionEntry(region.id, entry.id)}>
													<X class="h-3 w-3" />
												</Button>
											</div>
											<div class="space-y-1">
												<Label class="text-xs">Condition</Label>
												<AcvusEngineField
													mode="script"
													placeholder="e.g. @item.type == 'status'"
													value={entry.condition}
													oninput={(v) => updateRegionEntry(region.id, entry.id, { condition: v })}
													contextTypes={computeIterableEntryTypes(region.iterator, displayContextTypes)}
													{analysisErrors}
												/>
											</div>
											<div class="space-y-1">
												<Label class="text-xs">Template</Label>
												<AcvusEngineField
													mode="template"
													placeholder="Template content..."
													value={entry.template}
													oninput={(v) => updateRegionEntry(region.id, entry.id, { template: v })}
													contextTypes={computeIterableEntryTypes(region.iterator, displayContextTypes)}
													{analysisErrors}
												/>
											</div>
										</div>
									{/each}
									<Button variant="outline" size="sm" class="w-full border-dashed text-muted-foreground text-xs" onclick={() => addRegionEntry(region.id)}>
										<Plus class="h-3 w-3 mr-1" /> Add Entry
									</Button>
								</div>
							{/if}
						</div>
					{/each}

					<div class="flex gap-2">
						<Button variant="outline" size="sm" class="flex-1 border-dashed text-muted-foreground" onclick={() => addRegion('iterable')}>
							<Plus class="h-3 w-3 mr-1" /> Iterable Region
						</Button>
						<Button variant="outline" size="sm" class="flex-1 border-dashed text-muted-foreground" onclick={() => addRegion('static')}>
							<Plus class="h-3 w-3 mr-1" /> Static Region
						</Button>
					</div>
				</div>

				{#if bot.contextParams.length > 0}
					<Separator />

					<div class="space-y-3">
						<div>
							<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Unresolved Parameters</span>
							<p class="text-xs text-muted-foreground mt-1">Context refs across the full hierarchy (prompt + profile + bot) not provided by bindings or nodes.</p>
						</div>
						<ContextParamsEditor
							params={bot.contextParams}
							onupdate={handleParamsUpdate}
							onTypeChange={handleTypeChange}
							contextTypes={discoveredContextTypes}
							{analysisErrors}
						/>
					</div>
				{/if}
			</div>
		</ScrollArea>
	{:else}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
			No bot selected.
		</div>
	{/if}
</div>
