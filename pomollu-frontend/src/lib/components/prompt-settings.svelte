<script lang="ts">
	import type { ContextBinding, ContextParam, ParamOverride } from '$lib/types.js';
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

	import { type AnalysisOrchestrator, mergeParams, pruneOverrides } from '$lib/param-resolver.js';
	import { collectPromptDeps } from '$lib/dependencies.js';
	import { Download } from 'lucide-svelte';
	import { exportEntityZip } from '$lib/io.js';
	import { confirmDelete } from '$lib/confirm-dialog.svelte.js';
	import BasePage from './base-page.svelte';

	let { promptId, orchestrator }: { promptId: string; orchestrator: AnalysisOrchestrator } = $props();

	let prompt = $derived(promptStore.get(promptId));
	let deps = $derived(prompt ? collectPromptDeps(prompt) : []);
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
	let analysisResult = $state<ContextParam[]>([]);
	let mergedParams = $derived(mergeParams(analysisResult, prompt?.paramOverrides ?? {}));

	function runAnalysis() {
		if (!prompt) return;
		const result = orchestrator.analyzePrompt(prompt, (id) => providerStore.get(id)?.api);
		discoveredContextTypes = result.env.contextTypes;
		analysisResult = result.params;
		const pruned = pruneOverrides(prompt.paramOverrides, result.params);
		if (pruned) promptStore.update(promptId, (p) => ({ ...p, paramOverrides: pruned }));
	}

	function handleParamsUpdate(params: ContextParam[]) {
		const overrides: Record<string, ParamOverride> = {};
		for (const p of params) {
			overrides[p.name] = {
				resolution: p.resolution,
				...(p.userType ? { userType: p.userType } : {}),
				...(p.editorMode ? { editorMode: p.editorMode } : {}),
			};
		}
		promptStore.update(promptId, (p) => ({ ...p, paramOverrides: overrides }));
	}

	let dynamicParams = $derived(
		mergedParams.filter((p) => p.resolution.kind === 'dynamic').map((p) => p.name)
	);
</script>

<BasePage {deps} onConfigChange={runAnalysis} debounceMs={200}>
	<div class="flex items-center justify-between shrink-0 border-b px-3 py-2">
		<span class="text-sm font-medium">Prompt Settings</span>
		<div class="flex items-center gap-1">
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground" onclick={() => prompt && exportEntityZip(prompt, `asset_${prompt.id}`)} title="Export">
				<Download class="h-3.5 w-3.5" />
			</Button>
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={async () => { if (await confirmDelete('Delete this prompt?')) uiState.removePrompt(promptId); }} title="Delete prompt">
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
								docManager={orchestrator.docs.prompt}
								docKey={`prompt:binding:${binding.name}`}
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

				<Separator />

				<div class="space-y-3">
					<div>
						<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Assets</span>
						<p class="text-xs text-muted-foreground mt-1">Binary assets for this prompt. Accessible via <code class="text-[0.7rem]">asset_url("folder/name")</code>.</p>
					</div>
					<Button variant="outline" size="sm" onclick={() => uiState.openTab({ kind: 'assets', dbName: `asset_${prompt.id}`, entityName: prompt.name })}>
						Open Asset Editor
					</Button>
				</div>

				{#if mergedParams.length > 0}
					<Separator />

					<div class="space-y-3">
						<div>
							<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Unresolved Parameters</span>
							<p class="text-xs text-muted-foreground mt-1">Context refs detected in scripts that are not provided by bindings or nodes.</p>
						</div>
						<ContextParamsEditor
							params={mergedParams}
							onupdate={handleParamsUpdate}
							docManager={orchestrator.docs.prompt}
							level="prompt"
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
</BasePage>
