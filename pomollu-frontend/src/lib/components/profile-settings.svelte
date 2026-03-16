<script lang="ts">
	import type { ContextParam, ParamOverride } from '$lib/types.js';

	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Button } from '$lib/components/ui/button';
	import { Separator } from '$lib/components/ui/separator';
	import { profileStore, providerStore, uiState } from '$lib/stores.svelte.js';
	import ContextParamsEditor from './context-params-editor.svelte';

	import { type AnalysisOrchestrator, mergeParams, pruneOverrides } from '$lib/param-resolver.js';
	import { collectProfileDeps } from '$lib/dependencies.js';
	import { Download } from 'lucide-svelte';
	import { exportEntityZip } from '$lib/io.js';
	import { confirmDelete } from '$lib/confirm-dialog.svelte.js';
	import BasePage from './base-page.svelte';

	let { profileId, orchestrator }: { profileId: string; orchestrator: AnalysisOrchestrator } = $props();

	let profile = $derived(profileStore.get(profileId));

	// --- Unresolved params analysis ---

	let discoveredContextTypes = $state<Record<string, import('$lib/type-parser.js').TypeDesc>>({});
	let analysisResult = $state<ContextParam[]>([]);

	let deps = $derived(profile ? collectProfileDeps(profile) : []);
	let mergedParams = $derived(mergeParams(analysisResult, profile?.paramOverrides ?? {}));

	function runAnalysis() {
		if (!profile) return;
		const result = orchestrator.analyzeProfile(profile, (id) => providerStore.get(id)?.api);
		discoveredContextTypes = result.env.contextTypes;
		analysisResult = result.params;
		const pruned = pruneOverrides(profile.paramOverrides, result.params);
		if (pruned) profileStore.update(profileId, (p) => ({ ...p, paramOverrides: pruned }));
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
		profileStore.update(profileId, (p) => ({ ...p, paramOverrides: overrides }));
	}
</script>

<BasePage {deps} onConfigChange={runAnalysis} debounceMs={200}>
	<div class="flex items-center justify-between shrink-0 border-b px-3 py-2">
		<span class="text-sm font-medium">Profile Settings</span>
		<div class="flex items-center gap-1">
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground" onclick={() => profile && exportEntityZip(profile, `asset_${profile.id}`)} title="Export">
				<Download class="h-3.5 w-3.5" />
			</Button>
			<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={async () => { if (await confirmDelete('Delete this profile?')) uiState.removeProfile(profileId); }} title="Delete profile">
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

				<Separator />

				<div class="space-y-3">
					<div>
						<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Assets</span>
						<p class="text-xs text-muted-foreground mt-1">Binary assets for this profile. Accessible via <code class="text-[0.7rem]">asset_url("folder/name")</code>.</p>
					</div>
					<Button variant="outline" size="sm" onclick={() => uiState.openTab({ kind: 'assets', dbName: `asset_${profile.id}`, entityName: profile.name })}>
						Open Asset Editor
					</Button>
				</div>

				{#if mergedParams.length > 0}
					<Separator />

					<div class="space-y-3">
						<div>
							<span class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Unresolved Parameters</span>
							<p class="text-xs text-muted-foreground mt-1">Context refs detected in profile scripts. Prompt bindings are not visible here.</p>
						</div>
						<ContextParamsEditor
							params={mergedParams}
							onupdate={handleParamsUpdate}
							docManager={orchestrator.docs.profile}
							level="profile"
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
</BasePage>
