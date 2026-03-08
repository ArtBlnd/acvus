<script lang="ts">
	import type { SidebarNode, Prompt, Profile } from '$lib/types.js';
	import { providerStore, promptStore, profileStore, uiState, generateName } from '$lib/stores.svelte.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import SidebarList from './sidebar-list.svelte';
	import { pickJsonFile, withNewId, validatePrompt, validateProfile } from '$lib/io.js';

	let collapsed: Record<string, boolean> = $state({});

	function toggle(key: string) {
		collapsed[key] = !collapsed[key];
	}

	// --- Per-section tree: derived from store data ---
	let providerTree: SidebarNode[] = $derived(
		providerStore.providers.map((p): SidebarNode => ({ kind: 'item', id: p.id }))
	);
	let profileTree: SidebarNode[] = $derived(
		profileStore.profiles.map((p): SidebarNode => ({ kind: 'item', id: p.id }))
	);
	let promptTree: SidebarNode[] = $derived(
		promptStore.prompts.map((p): SidebarNode => ({ kind: 'item', id: p.id }))
	);

	// --- Add: create in store + append to tree ---
	function addProvider() {
		const name = generateName('New Provider', providerStore.providers.map(p => p.name));
		const p = providerStore.create(name);
		uiState.openProvider(p.id);
	}

	function addPrompt() {
		const name = generateName('New Prompt', promptStore.prompts.map(p => p.name));
		const p = promptStore.create(name);
		uiState.openPrompt(p.id);
	}

	async function importPrompt() {
		const data = await pickJsonFile();
		if (!validatePrompt(data)) { alert('Invalid prompt file'); return; }
		const p = withNewId(data);
		promptStore.import(p);
		uiState.openPrompt(p.id);
	}

	function addProfile() {
		const name = generateName('New Profile', profileStore.profiles.map(p => p.name));
		const p = profileStore.create(name);
		uiState.openProfile(p.id);
	}

	async function importProfile() {
		const data = await pickJsonFile();
		if (!validateProfile(data)) { alert('Invalid profile file'); return; }
		const p = withNewId(data);
		profileStore.import(p);
		uiState.openProfile(p.id);
	}

	// --- Delete: remove from tree + store ---
	function deleteProvider(id: string) {
		providerStore.remove(id);
		uiState.removeProvider(id);
	}

	function deletePrompt(id: string) {
		promptStore.remove(id);
		uiState.removePrompt(id);
	}

	function deleteProfile(id: string) {
		profileStore.remove(id);
		uiState.removeProfile(id);
	}

	// --- Active tab check ---
	function isActiveTab(kind: string, id: string): boolean {
		const tab = uiState.activeTab;
		if (!tab || tab.kind !== kind) return false;
		switch (tab.kind) {
			case 'provider': return tab.providerId === id;
			case 'profile': return tab.profileId === id;
			case 'prompt': return tab.promptId === id;
			default: return false;
		}
	}
</script>

<div class="flex h-full w-64 flex-col border-r">
	<div class="flex items-center border-b px-3 py-2.5">
		<span class="text-sm font-medium">Settings</span>
	</div>
	<ScrollArea class="flex-1">
		<div class="flex flex-col">
			<!-- Providers -->
			<div class="sidebar-section">
				<div class="sidebar-section-header">
					<button class="sidebar-section-toggle" onclick={() => toggle('providers')}>
						<svg class="sidebar-chevron" class:collapsed={collapsed['providers']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
						<span>Providers</span>
					</button>
				</div>
				{#if !collapsed['providers']}
					<SidebarList
						nodes={providerTree}
						onnodeschange={() => {}}
						onselect={(id) => { uiState.openProvider(id); uiState.closeMobileSidebars(); }}
						ondelete={deleteProvider}
						isActive={(id) => isActiveTab('provider', id)}
						itemLabel={(id) => providerStore.get(id)?.name ?? 'Provider'}
						onadd={addProvider}
					/>
				{/if}
			</div>

			<!-- Profiles -->
			<div class="sidebar-section">
				<div class="sidebar-section-header">
					<button class="sidebar-section-toggle" onclick={() => toggle('profiles')}>
						<svg class="sidebar-chevron" class:collapsed={collapsed['profiles']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
						<span>Profiles</span>
					</button>
					<button class="sidebar-import-btn" onclick={importProfile} title="Import profile">
						<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
					</button>
				</div>
				{#if !collapsed['profiles']}
					<SidebarList
						nodes={profileTree}
						onnodeschange={() => {}}
						onselect={(id) => { uiState.openProfile(id); uiState.closeMobileSidebars(); }}
						ondelete={deleteProfile}
						isActive={(id) => isActiveTab('profile', id)}
						itemLabel={(id) => profileStore.get(id)?.name ?? 'Profile'}
						onadd={addProfile}
					/>
				{/if}
			</div>

			<!-- Prompts -->
			<div class="sidebar-section">
				<div class="sidebar-section-header">
					<button class="sidebar-section-toggle" onclick={() => toggle('prompts')}>
						<svg class="sidebar-chevron" class:collapsed={collapsed['prompts']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
						<span>Prompts</span>
					</button>
					<button class="sidebar-import-btn" onclick={importPrompt} title="Import prompt">
						<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
					</button>
				</div>
				{#if !collapsed['prompts']}
					<SidebarList
						nodes={promptTree}
						onnodeschange={() => {}}
						onselect={(id) => { uiState.openPrompt(id); uiState.closeMobileSidebars(); }}
						ondelete={deletePrompt}
						isActive={(id) => isActiveTab('prompt', id)}
						itemLabel={(id) => promptStore.get(id)?.name ?? 'Prompt'}
						onadd={addPrompt}
					/>
				{/if}
			</div>
		</div>
	</ScrollArea>
</div>

<style>
	.sidebar-section {
		border-bottom: 1px solid var(--color-border);
	}
	.sidebar-section-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.5rem 0.375rem 0.625rem;
	}
	.sidebar-section-toggle {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		border: none;
		background: none;
		padding: 0;
		font-size: 0.75rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--color-foreground);
		cursor: pointer;
		user-select: none;
	}
	.sidebar-section-toggle:hover {
		color: var(--color-foreground);
	}
	.sidebar-chevron {
		width: 0.75rem;
		height: 0.75rem;
		transition: transform 0.15s;
		flex-shrink: 0;
	}
	.sidebar-chevron.collapsed {
		transform: rotate(-90deg);
	}
	.sidebar-import-btn {
		margin-left: auto;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 1.25rem;
		height: 1.25rem;
		border-radius: 0.25rem;
		border: none;
		background: transparent;
		color: var(--color-muted-foreground);
		cursor: pointer;
		opacity: 0.6;
		transition: opacity 0.15s, background 0.15s;
	}
	.sidebar-import-btn:hover {
		opacity: 1;
		background: var(--color-accent);
	}
</style>
