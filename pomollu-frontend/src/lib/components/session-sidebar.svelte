<script lang="ts">
	import { botStore, sessionStore, uiState } from '$lib/stores.svelte.js';
	import { Button } from '$lib/components/ui/button';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import SidebarItem from './sidebar-item.svelte';

	let activeBot = $derived(botStore.active);
	let sessions = $derived(activeBot ? sessionStore.forBot(activeBot.id) : []);

	function addSession() {
		if (!activeBot) return;
		uiState.createSession(activeBot.id, `Chat ${sessions.length + 1}`);
	}
</script>

<div class="flex h-full w-64 flex-col border-r">
	<div class="flex items-center justify-between border-b px-3 py-2.5">
		<span class="text-sm font-medium">Sessions</span>
		<Button variant="ghost" size="icon-sm" class="h-6 w-6" onclick={addSession} title="New chat">+</Button>
	</div>
	<ScrollArea class="flex-1">
		<div class="flex flex-col gap-0.5 p-2">
			{#each sessions as session (session.id)}
				<SidebarItem
					active={session.id === sessionStore.activeSessionId}
					onselect={() => { uiState.selectSession(session.id); uiState.closeMobileSidebars(); }}
					ondelete={() => uiState.removeSession(session.id)}
				>
					{session.name}
				</SidebarItem>
			{:else}
				<div class="px-2.5 py-2 text-xs text-muted-foreground/60">No sessions yet.</div>
			{/each}
		</div>
	</ScrollArea>
	<div class="border-t p-2">
		<button
			class="w-full rounded-md px-2.5 py-1.5 text-left text-sm transition-colors
				{uiState.activeTab?.kind === 'bot-settings'
				? 'bg-accent text-accent-foreground'
				: 'text-muted-foreground hover:bg-accent/50'}"
			onclick={() => { if (activeBot) { uiState.openBotSettings(activeBot.id); uiState.closeMobileSidebars(); } }}
		>
			Bot Settings
		</button>
	</div>
</div>
