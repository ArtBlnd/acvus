<script lang="ts">
	import type { RenderedCard } from '$lib/types.js';
	import { sanitizeHtml } from '$lib/sanitize.js';
	import { fly, fade } from 'svelte/transition';

	let { card, compact = false }: { card: RenderedCard; compact?: boolean } = $props();

	let sanitized = $derived(sanitizeHtml(card.content));
</script>

<div
	class="display-card"
	class:compact
	in:fly={{ y: 12, duration: 200, opacity: 0 }}
	out:fade={{ duration: 100 }}
>
	<div class="card-header">{card.name}</div>
	<div class="card-body">{@html sanitized}</div>
</div>

<style>
	.display-card {
		border: 1px solid var(--color-border);
		border-radius: 0.5rem;
		overflow: hidden;
	}
	.card-header {
		padding: 0.375rem 0.75rem;
		font-size: 0.6875rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--color-muted-foreground);
		background: var(--color-secondary);
		border-bottom: 1px solid var(--color-border);
	}
	.card-body {
		padding: 0.75rem;
		font-size: 0.875rem;
		line-height: 1.625;
		white-space: pre-wrap;
		color: var(--color-foreground);
	}
	.compact .card-header {
		padding: 0.25rem 0.5rem;
		font-size: 0.625rem;
	}
	.compact .card-body {
		padding: 0.5rem;
		font-size: 0.75rem;
		line-height: 1.5;
	}
</style>
