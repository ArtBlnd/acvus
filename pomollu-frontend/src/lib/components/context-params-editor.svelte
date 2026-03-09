<script lang="ts">
	import type { ContextParam } from '$lib/types.js';
	import type { TypeDesc, StructuredValue } from '$lib/type-parser.js';
	import { parseTypeDesc, isStructured, createDefaultValue, generateScript, isUnknownType, typeDescToString } from '$lib/type-parser.js';
	import { Input } from '$lib/components/ui/input';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { slide } from 'svelte/transition';
	import AcvusEngineField from './acvus-engine-field.svelte';
	import StructuredValueEditor from './structured-value-editor.svelte';

	let {
		params,
		onupdate,
		onTypeChange,
		contextTypes = {},
	}: {
		params: ContextParam[];
		onupdate: (params: ContextParam[]) => void;
		onTypeChange: (name: string, type: string) => void;
		contextTypes?: Record<string, TypeDesc>;
	} = $props();

	// Per-param structured value state (component-local, not persisted)
	let structuredValues = $state<Record<string, StructuredValue>>({});
	// Per-param raw mode override
	let rawModes = $state<Record<string, boolean>>({});
	// Track previous inferred types to detect changes
	let prevInferredTypes = $state<Record<string, string>>({});

	function setResolution(index: number, kind: 'static' | 'dynamic' | 'unresolved') {
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			if (kind === 'static') return { ...p, resolution: { kind: 'static' as const, value: '' } };
			if (kind === 'dynamic') return { ...p, resolution: { kind: 'dynamic' as const } };
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
		const updated = params.map((p, i) => {
			if (i !== index) return p;
			return { ...p, userType: parsed };
		});
		onupdate(updated);
		const param = params[index];
		if (param && type) {
			onTypeChange(param.name, type);
		}
		// Reset structured value when type changes
		if (param) {
			delete structuredValues[param.name];
		}
	}

	function resolvedTypeDesc(param: ContextParam): TypeDesc | undefined {
		return param.userType || (isUnknownType(param.inferredType) ? undefined : param.inferredType);
	}

	function getTypeDesc(param: ContextParam): TypeDesc | null {
		return resolvedTypeDesc(param) ?? null;
	}

	function shouldUseStructured(param: ContextParam): boolean {
		if (rawModes[param.name]) return false;
		const desc = getTypeDesc(param);
		return desc !== null && isStructured(desc);
	}

	// Reset structured values when inferredType changes
	$effect(() => {
		for (const param of params) {
			const typeKey = JSON.stringify(param.inferredType);
			const prev = prevInferredTypes[param.name];
			if (prev && prev !== typeKey) {
				delete structuredValues[param.name];
			}
			prevInferredTypes[param.name] = typeKey;
		}
	});

	// Initialize structured values lazily via $effect (not during render)
	$effect(() => {
		const pendingUpdates: { index: number; value: string }[] = [];
		for (let idx = 0; idx < params.length; idx++) {
			const param = params[idx];
			if (param.resolution.kind !== 'static') continue;
			if (rawModes[param.name]) continue;
			if (structuredValues[param.name]) continue;
			const desc = getTypeDesc(param);
			if (desc && isStructured(desc)) {
				const defaultVal = createDefaultValue(desc);
				structuredValues[param.name] = defaultVal;
				const script = generateScript(defaultVal, desc);
				pendingUpdates.push({ index: idx, value: script });
			}
		}
		if (pendingUpdates.length > 0) {
			const updated = params.map((p, i) => {
				const upd = pendingUpdates.find((u) => u.index === i);
				if (upd) return { ...p, resolution: { kind: 'static' as const, value: upd.value } };
				return p;
			});
			onupdate(updated);
		}
	});

	function getStructuredValue(param: ContextParam): StructuredValue {
		return structuredValues[param.name] ?? { kind: 'raw', script: '' };
	}

	function handleStructuredChange(index: number, param: ContextParam, value: StructuredValue) {
		structuredValues[param.name] = value;
		const desc = getTypeDesc(param);
		if (desc) {
			setStaticValue(index, generateScript(value, desc));
		}
	}

	function toggleRawMode(param: ContextParam) {
		rawModes[param.name] = !rawModes[param.name];
		if (!rawModes[param.name]) {
			delete structuredValues[param.name];
		}
	}
</script>

{#if params.length === 0}
	<p class="text-xs text-muted-foreground italic">No unresolved parameters.</p>
{:else}
	<div class="space-y-2">
		{#each params as param, i (param.name)}
			<div class="rounded-md border p-3 space-y-2" transition:slide={{ duration: 150 }}>
				<div class="flex items-center gap-2">
					<code class="text-xs font-semibold text-foreground">@{param.name}</code>
					{#if !isUnknownType(param.inferredType)}
						<Badge variant="secondary" class="text-[0.625rem]">{typeDescToString(param.inferredType)}</Badge>
					{:else if param.userType}
						<Badge variant="outline" class="text-[0.625rem]">{typeDescToString(param.userType)}</Badge>
					{:else}
						<Badge variant="destructive" class="text-[0.625rem]">?</Badge>
					{/if}
					<div class="flex-1"></div>
					<div class="flex gap-1">
						{#if param.resolution.kind === 'static' && getTypeDesc(param) && isStructured(getTypeDesc(param)!)}
							<Button
								variant="ghost"
								size="sm"
								class="h-6 text-[0.5625rem] px-1.5 text-muted-foreground"
								onclick={() => toggleRawMode(param)}
							>{rawModes[param.name] ? 'structured' : 'raw'}</Button>
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

				{#if isUnknownType(param.inferredType) && !param.userType}
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
							{contextTypes}
						/>
					{:else}
						<AcvusEngineField
							mode="script"
							value={param.resolution.value}
							oninput={(v) => setStaticValue(i, v)}
							placeholder="static value expression..."
							{contextTypes}
							expectedTailType={resolvedTypeDesc(param)}
						/>
					{/if}
					{#if !param.resolution.value.trim()}
						<p class="text-[0.625rem] text-destructive">Static value is required.</p>
					{/if}
				{:else if param.resolution.kind === 'dynamic'}
					<p class="text-[0.625rem] text-muted-foreground italic">Provided at each turn input.</p>
				{/if}
			</div>
		{/each}
	</div>
{/if}
