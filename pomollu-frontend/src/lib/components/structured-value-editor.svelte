<script lang="ts">
	import type { TypeDesc, StructuredValue } from '$lib/type-parser.js';
	import { createDefaultValue, isStructured } from '$lib/type-parser.js';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import * as Select from '$lib/components/ui/select';
	import AcvusEngineField from './acvus-engine-field.svelte';
	import Self from './structured-value-editor.svelte';

	const MAX_DEPTH = 5;

	let {
		typeDesc,
		value,
		onchange,
		contextTypes = {},
		depth = 0,
	}: {
		typeDesc: TypeDesc;
		value: StructuredValue;
		onchange: (value: StructuredValue) => void;
		contextTypes?: Record<string, TypeDesc>;
		depth?: number;
	} = $props();

	// Fall back to raw editor if too deep or unsupported
	let useRaw = $derived(
		depth >= MAX_DEPTH || typeDesc.kind === 'list' || typeDesc.kind === 'unsupported'
	);
</script>

{#if useRaw}
	<AcvusEngineField
		mode="script"
		value={value.kind === 'raw' ? value.script : ''}
		oninput={(v) => onchange({ kind: 'raw', script: v })}
		{contextTypes}
	/>
{:else if typeDesc.kind === 'primitive'}
	{#if typeDesc.name === 'Bool'}
		<div class="flex items-center gap-2">
			<Switch
				checked={value.kind === 'primitive' && value.value === 'true'}
				onCheckedChange={(checked) => onchange({ kind: 'primitive', value: checked ? 'true' : 'false' })}
			/>
			<span class="text-xs text-muted-foreground">{value.kind === 'primitive' ? value.value : 'false'}</span>
		</div>
	{:else if typeDesc.name === 'Int'}
		<Input
			type="number"
			step="1"
			class="text-xs h-7"
			value={value.kind === 'primitive' ? value.value : '0'}
			oninput={(e) => onchange({ kind: 'primitive', value: e.currentTarget.value })}
		/>
	{:else if typeDesc.name === 'Float'}
		<Input
			type="number"
			step="any"
			class="text-xs h-7"
			value={value.kind === 'primitive' ? value.value : '0.0'}
			oninput={(e) => onchange({ kind: 'primitive', value: e.currentTarget.value })}
		/>
	{:else}
		<!-- String -->
		<Input
			class="text-xs h-7"
			value={value.kind === 'primitive' ? value.value : ''}
			oninput={(e) => onchange({ kind: 'primitive', value: e.currentTarget.value })}
		/>
	{/if}

{:else if typeDesc.kind === 'option'}
	{@const isNone = value.kind === 'option-none'}
	<div class="space-y-1.5">
		<Select.Root
			type="single"
			value={isNone ? 'None' : 'Some'}
			onValueChange={(v) => {
				if (v === 'None') {
					onchange({ kind: 'option-none' });
				} else {
					onchange({
						kind: 'option-some',
						inner: value.kind === 'option-some' ? value.inner : createDefaultValue(typeDesc.inner),
					});
				}
			}}
		>
			<Select.Trigger class="h-7 text-xs w-full" size="sm">
				<span>{isNone ? 'None' : 'Some'}</span>
			</Select.Trigger>
			<Select.Content>
				<Select.Item value="Some">Some</Select.Item>
				<Select.Item value="None">None</Select.Item>
			</Select.Content>
		</Select.Root>
		{#if !isNone}
			{#if isStructured(typeDesc.inner)}
				<div class="pl-3 border-l-2 border-muted">
					<Self
						typeDesc={typeDesc.inner}
						value={value.kind === 'option-some' ? value.inner : createDefaultValue(typeDesc.inner)}
						onchange={(v: StructuredValue) => onchange({ kind: 'option-some', inner: v })}
						{contextTypes}
						depth={depth + 1}
					/>
				</div>
			{:else}
				<div class="pl-3 border-l-2 border-muted">
					<AcvusEngineField
						mode="script"
						value={value.kind === 'option-some' && value.inner.kind === 'raw' ? value.inner.script : ''}
						oninput={(v) => onchange({ kind: 'option-some', inner: { kind: 'raw', script: v } })}
						{contextTypes}
					/>
				</div>
			{/if}
		{/if}
	</div>

{:else if typeDesc.kind === 'enum'}
	{@const selected = value.kind === 'enum-variant' ? value.tag : (typeDesc.variants[0]?.tag ?? '')}
	{@const selectedVariant = typeDesc.variants.find((v) => v.tag === selected)}
	<div class="space-y-1.5">
		<Select.Root
			type="single"
			value={selected}
			onValueChange={(tag) => {
				const variant = typeDesc.kind === 'enum' ? typeDesc.variants.find((v) => v.tag === tag) : undefined;
				const payload = variant?.hasPayload && variant.payloadType
					? createDefaultValue(variant.payloadType)
					: undefined;
				onchange({ kind: 'enum-variant', tag, payload });
			}}
		>
			<Select.Trigger class="h-7 text-xs w-full" size="sm">
				<span>{selected}</span>
			</Select.Trigger>
			<Select.Content>
				{#each typeDesc.variants as variant (variant.tag)}
					<Select.Item value={variant.tag}>{variant.tag}</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
		{#if selectedVariant?.hasPayload && selectedVariant.payloadType}
			{@const payloadType = selectedVariant.payloadType}
			{@const payloadValue = value.kind === 'enum-variant' && value.payload ? value.payload : createDefaultValue(payloadType)}
			{#if isStructured(payloadType)}
				<div class="pl-3 border-l-2 border-muted">
					<Self
						typeDesc={payloadType}
						value={payloadValue}
						onchange={(v: StructuredValue) => onchange({ kind: 'enum-variant', tag: selected, payload: v })}
						{contextTypes}
						depth={depth + 1}
					/>
				</div>
			{:else}
				<div class="pl-3 border-l-2 border-muted">
					<AcvusEngineField
						mode="script"
						value={payloadValue.kind === 'raw' ? payloadValue.script : ''}
						oninput={(v) => onchange({ kind: 'enum-variant', tag: selected, payload: { kind: 'raw', script: v } })}
						{contextTypes}
					/>
				</div>
			{/if}
		{/if}
	</div>

{:else if typeDesc.kind === 'object'}
	<div class="space-y-2">
		{#each typeDesc.fields as field (field.name)}
			<div class="space-y-0.5">
				<Label class="text-[0.625rem] text-muted-foreground font-medium">{field.name}</Label>
				{#if isStructured(field.type)}
					<Self
						typeDesc={field.type}
						value={value.kind === 'object' ? (value.fields[field.name] ?? createDefaultValue(field.type)) : createDefaultValue(field.type)}
						onchange={(v: StructuredValue) => {
							const fields = value.kind === 'object' ? { ...value.fields } : {};
							fields[field.name] = v;
							onchange({ kind: 'object', fields });
						}}
						{contextTypes}
						depth={depth + 1}
					/>
				{:else}
					<AcvusEngineField
						mode="script"
						value={value.kind === 'object' && value.fields[field.name]?.kind === 'raw'
							? (value.fields[field.name] as { kind: 'raw'; script: string }).script
							: ''}
						oninput={(v) => {
							const fields = value.kind === 'object' ? { ...value.fields } : {};
							fields[field.name] = { kind: 'raw', script: v };
							onchange({ kind: 'object', fields });
						}}
						{contextTypes}
					/>
				{/if}
			</div>
		{/each}
	</div>
{/if}
