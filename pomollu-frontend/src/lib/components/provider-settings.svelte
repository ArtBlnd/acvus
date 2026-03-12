<script lang="ts">
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Button } from '$lib/components/ui/button';
	import * as Select from '$lib/components/ui/select';
	import type { ApiKind } from '$lib/types.js';
	import { providerStore, uiState } from '$lib/stores.svelte.js';
	import type { EntityRef } from '$lib/entity-versions.svelte.js';
	import BasePage from './base-page.svelte';

	let { providerId }: { providerId: string } = $props();

	let provider = $derived(providerStore.get(providerId));
	let deps = $derived<EntityRef[]>([{ kind: 'provider', id: providerId }]);
</script>

<BasePage {deps} onConfigChange={() => {}}>
<div class="flex h-full flex-col">
	<div class="flex items-center justify-between shrink-0 border-b px-4 py-2">
		<span class="text-sm font-medium">Provider Settings</span>
		<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={() => uiState.removeProvider(providerId)} title="Delete provider">
			&times;
		</Button>
	</div>

	{#if provider}
		<ScrollArea class="flex-1">
			<div class="space-y-4 px-6 py-10">
				<div class="space-y-1">
					<Label>Name</Label>
					<Input
						value={provider.name}
						oninput={(e) => providerStore.update(provider.id, (p) => ({ ...p, name: e.currentTarget.value }))}
					/>
				</div>
				<div class="space-y-1">
					<Label>API Type</Label>
					<Select.Root type="single" value={provider.api} onValueChange={(v) => providerStore.update(provider.id, (p) => ({ ...p, api: v as ApiKind }))}>
						<Select.Trigger class="w-full">
							{provider.api === 'openai' ? 'OpenAI' : provider.api === 'anthropic' ? 'Anthropic' : 'Google'}
						</Select.Trigger>
						<Select.Content>
							<Select.Item value="openai">OpenAI</Select.Item>
							<Select.Item value="anthropic">Anthropic</Select.Item>
							<Select.Item value="google">Google</Select.Item>
						</Select.Content>
					</Select.Root>
				</div>
				<div class="space-y-1">
					<Label>Endpoint</Label>
					<Input
						value={provider.endpoint}
						oninput={(e) => providerStore.update(provider.id, (p) => ({ ...p, endpoint: e.currentTarget.value }))}
						placeholder="https://..."
					/>
				</div>
				<div class="space-y-1">
					<Label>API Key</Label>
					<Input
						type="password"
						value={provider.apiKey}
						oninput={(e) => providerStore.update(provider.id, (p) => ({ ...p, apiKey: e.currentTarget.value }))}
						placeholder="sk-..."
					/>
				</div>
			</div>
		</ScrollArea>
	{:else}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
			No provider selected.
		</div>
	{/if}
</div>
</BasePage>
