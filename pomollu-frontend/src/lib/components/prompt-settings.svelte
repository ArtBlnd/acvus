<script lang="ts">
	import type { ContextBinding, ContextParam } from '$lib/types.js';
	import { HISTORY_BINDING_NAME, HISTORY_ENTRY_TYPE, CONTEXT_TYPE } from '$lib/types.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Button } from '$lib/components/ui/button';
	import { Separator } from '$lib/components/ui/separator';
	import { Plus, X, Lock } from 'lucide-svelte';
	import { slide } from 'svelte/transition';
	import { promptStore, providerStore, uiState } from '$lib/stores.svelte.js';
	import * as Select from '$lib/components/ui/select';
	import AcvusEngineField from './acvus-engine-field.svelte';
	import ContextParamsEditor from './context-params-editor.svelte';
	import {
		collectScriptsFromBindings,
		collectScriptsFromTree,
		collectNodeNames,
		analyzeLevel,
		mergeDiscoveredParams,
	} from '$lib/param-resolver.js';
	import { Download } from 'lucide-svelte';
	import { downloadJson } from '$lib/io.js';
	import { onDestroy } from 'svelte';

	let { promptId }: { promptId: string } = $props();

	let prompt = $derived(promptStore.get(promptId));
	let bindings = $derived(prompt?.contextBindings ?? []);

	function isRequired(binding: ContextBinding): boolean {
		return binding.name === HISTORY_BINDING_NAME;
	}

	function addBinding() {
		if (!prompt) return;
		promptStore.update(promptId, (p) => ({
			...p,
			contextBindings: [...p.contextBindings, { name: '', script: '' }]
		}));
	}

	function updateBinding(index: number, patch: Partial<ContextBinding>) {
		if (!prompt) return;
		promptStore.update(promptId, (p) => ({
			...p,
			contextBindings: p.contextBindings.map((b, i) => i === index ? { ...b, ...patch } : b)
		}));
	}

	function removeBinding(index: number) {
		if (!prompt) return;
		promptStore.update(promptId, (p) => ({
			...p,
			contextBindings: p.contextBindings.filter((_, i) => i !== index)
		}));
	}

	let hasDuplicateBindingName = $derived.by(() => {
		const names = bindings.map((b) => b.name).filter((n) => n !== '');
		return new Set(names).size !== names.length;
	});

	function isDuplicateBinding(index: number): boolean {
		const name = bindings[index]?.name;
		if (!name) return false;
		return bindings.some((b, i) => i !== index && b.name === name);
	}

	// --- Unresolved params analysis ---

	let discoveredContextTypes = $state<Record<string, import('$lib/type-parser.js').TypeDesc>>({});
	let analyzeTimer: ReturnType<typeof setTimeout> | null = null;

	function runAnalysis() {
		if (!prompt) return;
		const nodeNames = collectNodeNames(prompt.children);
		nodeNames.add('context');
		const baseTypes: Record<string, import('$lib/type-parser.js').TypeDesc> = { context: CONTEXT_TYPE };
		const { discoveredTypes, unresolvedKeys } = analyzeLevel({
			scripts: [
				...collectScriptsFromBindings(prompt.contextBindings),
				...collectScriptsFromTree(prompt.children),
			],
			nodeNames,
			providedKeys: new Set(prompt.contextBindings.map((b) => b.name).filter((n) => n)),
			existingParams: prompt.contextParams,
			baseTypes,
			children: prompt.children,
			getApi: (id) => providerStore.get(id)?.api ?? 'openai',
		});
		discoveredContextTypes = discoveredTypes;
		promptStore.update(promptId, (p) => ({
			...p, contextParams: mergeDiscoveredParams(p.contextParams, unresolvedKeys)
		}));
	}

	function scheduleAnalysis() {
		if (analyzeTimer) clearTimeout(analyzeTimer);
		analyzeTimer = setTimeout(() => {
			analyzeTimer = null;
			runAnalysis();
		}, 200);
	}

	// Derive a stable key from binding scripts + children structure.
	// Only re-analyze when actual script content changes, not contextParams.
	let analysisKey = $derived(JSON.stringify([
		prompt?.contextBindings.map((b) => [b.name, b.script]),
		prompt?.children,
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
		promptStore.update(promptId, (p) => ({ ...p, contextParams: params }));
	}

	function handleTypeChange(_name: string, _type: string) {
		scheduleAnalysis();
	}

	let dynamicParams = $derived(
		(prompt?.contextParams ?? []).filter((p) => p.resolution.kind === 'dynamic').map((p) => p.name)
	);

	let locked = $derived(uiState.isPromptBusy(promptId));
</script>

<div class="flex h-full flex-col" class:pointer-events-none={locked} class:opacity-60={locked}>
	{#if locked}
		<div class="shrink-0 border-b bg-amber-500/10 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">Turn in progress — editing locked</div>
	{/if}
	<div class="flex items-center justify-between shrink-0 border-b px-4 py-2">
		<span class="text-sm font-medium">Prompt Settings</span>
		<div class="flex items-center gap-1">
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground" onclick={() => prompt && downloadJson(prompt, `${prompt.name}.prompt.json`)} title="Export">
				<Download class="h-3.5 w-3.5" />
			</Button>
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={() => uiState.removePrompt(promptId)} title="Delete prompt">
				&times;
			</Button>
		</div>
	</div>

	{#if prompt}
		<ScrollArea class="flex-1">
			<div class="mx-auto max-w-2xl space-y-4 px-6 py-10">
				<div class="space-y-1">
					<Label>Prompt Name</Label>
					<Input
						value={prompt.name}
						oninput={(e) => promptStore.update(prompt.id, (p) => ({ ...p, name: e.currentTarget.value }))}
					/>
				</div>

				<Separator />

				<div class="space-y-3">
					<div>
						<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Context Bindings</span>
						<p class="text-xs text-muted-foreground mt-1">Bind context variables (<code class="text-[0.7rem]">@name</code>) via script. Type is inferred automatically.</p>
					</div>

					{#each bindings as binding, i (i)}
						{@const required = isRequired(binding)}
						{@const duplicate = isDuplicateBinding(i)}
						<div class="rounded-md border p-3 space-y-2" transition:slide={{ duration: 150 }}>
							<div class="flex items-center gap-2">
								{#if required}
									<Lock class="h-3.5 w-3.5 text-muted-foreground shrink-0" />
								{/if}
								<Input
									class="flex-1 text-sm {duplicate ? 'border-destructive focus-visible:ring-destructive' : ''}"
									value={binding.name}
									oninput={(e) => updateBinding(i, { name: e.currentTarget.value })}
									placeholder="name..."
									disabled={required}
								/>
								{#if !required}
									<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive shrink-0" onclick={() => removeBinding(i)}>
										<X class="h-3.5 w-3.5" />
									</Button>
								{/if}
							</div>
							<AcvusEngineField
								mode="script"
								value={binding.script}
								oninput={(v) => updateBinding(i, { script: v })}
								placeholder="e.g. @messages | map(...)"
								contextTypes={discoveredContextTypes}
								expectedTailType={binding.name === HISTORY_BINDING_NAME ? HISTORY_ENTRY_TYPE : undefined}
							/>
							{#if binding.name === HISTORY_BINDING_NAME}
								<p class="text-xs text-muted-foreground">Expected: <code class="text-[0.65rem]">List&lt;&#123;content: String, content_type: String, role: String&#125;&gt;</code></p>
							{/if}
						</div>
					{/each}

					{#if hasDuplicateBindingName}
						<p class="text-xs text-destructive">Duplicate binding names found.</p>
					{/if}

					<Button variant="outline" size="sm" class="w-full border-dashed text-muted-foreground" onclick={addBinding}>
						<Plus class="h-3 w-3 mr-1" /> Add Binding
					</Button>
				</div>

				{#if prompt.contextParams.length > 0}
					<Separator />

					<div class="space-y-3">
						<div>
							<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Unresolved Parameters</span>
							<p class="text-xs text-muted-foreground mt-1">Context refs detected in scripts that are not provided by bindings or nodes.</p>
						</div>
						<ContextParamsEditor
							params={prompt.contextParams}
							onupdate={handleParamsUpdate}
							onTypeChange={handleTypeChange}
							contextTypes={discoveredContextTypes}
						/>
					</div>
				{/if}

				{#if dynamicParams.length > 0}
					<Separator />

					<div class="space-y-2">
						<div>
							<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Chat Input</span>
							<p class="text-xs text-muted-foreground mt-1">Which dynamic parameter receives the user's chat input each turn.</p>
						</div>
						<Select.Root
							type="single"
							value={prompt.inputParam ?? ''}
							onValueChange={(v) => promptStore.update(promptId, (p) => ({ ...p, inputParam: v }))}
						>
							<Select.Trigger class="w-full">
								{#if prompt.inputParam}
									@{prompt.inputParam}
								{:else}
									<span class="text-muted-foreground">Select...</span>
								{/if}
							</Select.Trigger>
							<Select.Content>
								{#each dynamicParams as name (name)}
									<Select.Item value={name}>@{name}</Select.Item>
								{/each}
							</Select.Content>
						</Select.Root>
						{#if !prompt.inputParam}
							<p class="text-[0.625rem] text-destructive">Select a chat input parameter.</p>
						{/if}
					</div>
				{/if}
			</div>
		</ScrollArea>
	{:else}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
			No prompt selected.
		</div>
	{/if}
</div>
