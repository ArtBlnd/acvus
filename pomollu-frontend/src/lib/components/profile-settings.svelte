<script lang="ts">
	import type { ContextParam } from '$lib/types.js';
	import { CONTEXT_TYPE } from '$lib/types.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Button } from '$lib/components/ui/button';
	import { Separator } from '$lib/components/ui/separator';
	import { profileStore, providerStore, uiState } from '$lib/stores.svelte.js';
	import ContextParamsEditor from './context-params-editor.svelte';
	import {
		collectScriptsFromTree,
		collectNodeNames,
		analyzeLevel,
		mergeDiscoveredParams,
	} from '$lib/param-resolver.js';
	import { Download } from 'lucide-svelte';
	import { downloadJson } from '$lib/io.js';
	import { onDestroy } from 'svelte';

	let { profileId }: { profileId: string } = $props();

	let profile = $derived(profileStore.get(profileId));

	// --- Unresolved params analysis ---

	let analyzeTimer: ReturnType<typeof setTimeout> | null = null;
	let discoveredContextTypes = $state<Record<string, import('$lib/type-parser.js').TypeDesc>>({});

	function runAnalysis() {
		if (!profile) return;
		const nodeNames = collectNodeNames(profile.children);
		nodeNames.add('context');
		const baseTypes: Record<string, import('$lib/type-parser.js').TypeDesc> = { context: CONTEXT_TYPE };
		const { discoveredTypes, unresolvedKeys } = analyzeLevel({
			scripts: collectScriptsFromTree(profile.children),
			nodeNames,
			providedKeys: new Set(),
			existingParams: profile.contextParams,
			baseTypes,
			children: profile.children,
			getApi: (id) => providerStore.get(id)?.api ?? 'openai',
		});
		discoveredContextTypes = discoveredTypes;
		profileStore.update(profileId, (p) => ({
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

	let analysisKey = $derived(JSON.stringify(profile?.children));

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
		profileStore.update(profileId, (p) => ({ ...p, contextParams: params }));
	}

	function handleTypeChange(_name: string, _type: string) {
		scheduleAnalysis();
	}

	let locked = $derived(uiState.isProfileBusy(profileId));
</script>

<div class="flex h-full flex-col" class:pointer-events-none={locked} class:opacity-60={locked}>
	{#if locked}
		<div class="shrink-0 border-b bg-amber-500/10 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">Turn in progress — editing locked</div>
	{/if}
	<div class="flex items-center justify-between shrink-0 border-b px-4 py-2">
		<span class="text-sm font-medium">Profile Settings</span>
		<div class="flex items-center gap-1">
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground" onclick={() => profile && downloadJson(profile, `${profile.name}.profile.json`)} title="Export">
				<Download class="h-3.5 w-3.5" />
			</Button>
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={() => uiState.removeProfile(profileId)} title="Delete profile">
				&times;
			</Button>
		</div>
	</div>

	{#if profile}
		<ScrollArea class="flex-1">
			<div class="space-y-4 px-6 py-10">
				<div class="space-y-1">
					<Label>Profile Name</Label>
					<Input
						value={profile.name}
						oninput={(e) => profileStore.update(profile.id, (p) => ({ ...p, name: e.currentTarget.value }))}
					/>
				</div>

				{#if profile.contextParams.length > 0}
					<Separator />

					<div class="space-y-3">
						<div>
							<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Unresolved Parameters</span>
							<p class="text-xs text-muted-foreground mt-1">Context refs detected in profile scripts. Prompt bindings are not visible here.</p>
						</div>
						<ContextParamsEditor
							params={profile.contextParams}
							onupdate={handleParamsUpdate}
							onTypeChange={handleTypeChange}
							contextTypes={discoveredContextTypes}
						/>
					</div>
				{/if}
			</div>
		</ScrollArea>
	{:else}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
			No profile selected.
		</div>
	{/if}
</div>
