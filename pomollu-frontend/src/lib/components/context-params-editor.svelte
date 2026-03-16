<script lang="ts">
	import type { ContextParam } from '$lib/types.js';
	import type { TypeDesc, StructuredValue } from '$lib/type-parser.js';
	import type { DocumentManager } from '$lib/document-manager.svelte.js';
	import { parseTypeDesc, parseScript, isStructured, createDefaultValue, generateScript, isUnknownType, typeDescToString } from '$lib/type-parser.js';
	import { Input } from '$lib/components/ui/input';
	import { Button } from '$lib/components/ui/button';
	import { slide } from 'svelte/transition';
	import AcvusEngineField from './acvus-engine-field.svelte';
	import StructuredValueEditor from './structured-value-editor.svelte';

	let {
		params,
		onupdate,
		onTypeChange,
		docManager,
		level,
	}: {
		params: ContextParam[];
		onupdate: (params: ContextParam[]) => void;
		onTypeChange?: (name: string, type: string) => void;
		docManager?: DocumentManager;
		level?: string;
	} = $props();

	// Component-local cache for in-flight structured edits.
	// Only populated when the user actively edits via StructuredValueEditor.
	// On remount, values are re-derived from stored script via parseScript (no cache needed).
	let editCache = $state<Record<string, StructuredValue>>({});

	function resolvedTypeDesc(param: ContextParam): TypeDesc | undefined {
		return param.userType || (isUnknownType(param.inferredType) ? undefined : param.inferredType);
	}

	function getTypeDesc(param: ContextParam): TypeDesc | null {
		return resolvedTypeDesc(param) ?? null;
	}

	function isRawMode(param: ContextParam): boolean {
		return param.editorMode === 'raw';
	}

	function shouldUseStructured(param: ContextParam): boolean {
		if (isRawMode(param)) return false;
		const desc = getTypeDesc(param);
		return desc !== null && isStructured(desc);
	}

	/** Get structured value: prefer edit cache, then parse from stored script. */
	function getStructuredValue(param: ContextParam): StructuredValue {
		const cached = editCache[param.name];
		if (cached) return cached;

		// Parse the stored script back into a StructuredValue (pure, no state mutation).
		if (param.resolution.kind === 'static' && param.resolution.value.trim()) {
			const desc = getTypeDesc(param);
			if (desc) {
				const parsed = parseScript(param.resolution.value, desc);
				if (parsed) return parsed;
			}
		}
		return { kind: 'raw', script: '' };
	}

	// --- Actions ---

	function setResolution(index: number, kind: 'static' | 'dynamic' | 'unresolved') {
		const param = params[index];
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			if (kind === 'static') {
				// Generate default value for structured types when switching to static.
				const desc = getTypeDesc(p);
				if (desc && isStructured(desc)) {
					const defaultVal = createDefaultValue(desc);
					editCache[p.name] = defaultVal;
					return {
						...p,
						resolution: { kind: 'static' as const, value: generateScript(defaultVal, desc) },
						editorMode: 'structured' as const,
					};
				}
				return { ...p, resolution: { kind: 'static' as const, value: '' } };
			}
			if (kind === 'dynamic') {
				delete editCache[p.name];
				return { ...p, resolution: { kind: 'dynamic' as const } };
			}
			delete editCache[p.name];
			return { ...p, resolution: { kind: 'unresolved' as const } };
		});
		onupdate(updated);
	}

	function setStaticValue(index: number, value: string) {
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			return { ...p, resolution: { kind: 'static' as const, value } };
		});
		onupdate(updated);
	}

	function setUserType(index: number, type: string) {
		const parsed = type ? parseTypeDesc(type) : undefined;
		const valid = parsed && parsed.kind !== 'unsupported' ? parsed : undefined;
		const param = params[index];
		if (param) delete editCache[param.name];
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			return { ...p, userType: valid };
		});
		onupdate(updated);
		if (param && valid) {
			onTypeChange?.(param.name, type);
		}
	}

	function handleStructuredChange(index: number, param: ContextParam, value: StructuredValue) {
		editCache[param.name] = value;
		const desc = getTypeDesc(param);
		if (desc) {
			setStaticValue(index, generateScript(value, desc));
		}
	}

	function toggleRawMode(param: ContextParam) {
		const newMode = isRawMode(param) ? 'structured' as const : 'raw' as const;
		if (newMode === 'structured') {
			// Switching to structured: clear cache so it re-parses from stored value.
			delete editCache[param.name];
		}
		const updated = params.map((p) => {
			if (p.name !== param.name) return p;
			return { ...p, editorMode: newMode };
		});
		onupdate(updated);
	}

	/** Type string for placeholder display. Empty if unknown and no user hint. */
	function typePlaceholder(param: ContextParam): string {
		const desc = resolvedTypeDesc(param);
		return desc ? typeDescToString(desc) : '';
	}

	let activeParams = $derived(params.filter((p) => p.active !== false));
</script>

{#if activeParams.length === 0}
	<p class="text-xs text-muted-foreground italic">No unresolved parameters.</p>
{:else}
	<div class="space-y-2">
		{#each activeParams as param (param.name)}
			{@const i = params.indexOf(param)}
			<div class="rounded-md border p-3 space-y-2 {param.resolution.kind === 'unresolved' ? 'border-destructive' : ''}" transition:slide={{ duration: 150 }}>
				<div class="flex flex-wrap items-center gap-2">
					<code class="text-xs font-semibold text-foreground break-all">@{param.name}</code>
					<div class="flex-1"></div>
					<div class="flex gap-1 shrink-0">
						{#if param.resolution.kind === 'static' && getTypeDesc(param) && isStructured(getTypeDesc(param)!)}
							<Button
								variant="ghost"
								size="sm"
								class="h-6 text-[0.5625rem] px-1.5 text-muted-foreground"
								onclick={() => toggleRawMode(param)}
							>{isRawMode(param) ? 'structured' : 'raw'}</Button>
						{/if}
						<Button
							variant={param.resolution.kind === 'static' ? 'default' : 'outline'}
							size="sm"
							class="h-6 text-[0.625rem] px-2"
							onclick={() => setResolution(i, 'static')}
						>Static</Button>
						<Button
							variant={param.resolution.kind === 'dynamic' ? 'default' : 'outline'}
							size="sm"
							class="h-6 text-[0.625rem] px-2"
							onclick={() => setResolution(i, 'dynamic')}
						>Dynamic</Button>
					</div>
				</div>

				{#if isUnknownType(param.inferredType)}
					<div class="space-y-1">
						<span class="text-[0.625rem] text-muted-foreground">Type hint</span>
						<Input
							class="text-xs h-7"
							placeholder="e.g. String, Int, List<String>..."
							value={param.userType ? typeDescToString(param.userType) : ''}
							oninput={(e) => setUserType(i, e.currentTarget.value)}
						/>
					</div>
				{/if}

				{#if param.resolution.kind === 'static'}
					{#if shouldUseStructured(param)}
						{@const desc = getTypeDesc(param)!}
						<StructuredValueEditor
							typeDesc={desc}
							value={getStructuredValue(param)}
							onchange={(v) => handleStructuredChange(i, param, v)}
							{docManager}
							docKey={docManager && level ? `${level}:param:${param.name}:struct` : undefined}
						/>
					{:else}
						<AcvusEngineField
							mode="script"
							value={param.resolution.value}
							oninput={(v) => setStaticValue(i, v)}
							placeholder={typePlaceholder(param) || 'static value expression...'}
							{docManager}
							docKey={docManager && level ? `${level}:param:${param.name}:static` : undefined}
						/>
					{/if}
					{#if !param.resolution.value.trim()}
						<p class="text-[0.625rem] text-destructive">Static value is required.</p>
					{:else if !resolvedTypeDesc(param)}
						<p class="text-[0.625rem] text-destructive">Type is unknown. Specify a type hint.</p>
					{/if}
				{:else if param.resolution.kind === 'dynamic'}
					<p class="text-[0.625rem] text-muted-foreground italic">Provided at each turn input.</p>
				{/if}
			</div>
		{/each}
	</div>
{/if}
