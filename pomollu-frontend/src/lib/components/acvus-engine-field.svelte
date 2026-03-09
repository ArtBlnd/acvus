<script lang="ts">
	import { onDestroy } from 'svelte';
	import { typecheckWithTypes, typecheckWithTail, analyzeWithTypes } from '$lib/engine.js';
	import type { TypeDesc } from '$lib/type-parser.js';
	import { isUnknownType } from '$lib/type-parser.js';
	import { highlightTemplate, highlightScript } from '$lib/highlight.js';

	let {
		value,
		oninput,
		mode = 'script',
		placeholder = '',
		rows = 1,
		unlimited = false,
		contextTypes = {},
		expectedTailType,
		// 2-pass typecheck: analyze (lenient) first to discover unresolved context keys,
		// then hard typecheck with merged context. Only for node-internal fields
		// (strategy key/history bind, initial value, assert, messages, iterators).
		// Do NOT enable for context bindings, display entries, or regions.
		discoverContext = false,
	}: {
		value: string;
		oninput: (value: string) => void;
		mode?: 'script' | 'template';
		placeholder?: string;
		rows?: number;
		unlimited?: boolean;
		contextTypes?: Record<string, TypeDesc>;
		expectedTailType?: TypeDesc;
		discoverContext?: boolean;
	} = $props();

	let error = $state('');
	let debounceTimer: ReturnType<typeof setTimeout> | null = null;
	let hlHtml = $state('');
	let textareaEl = $state<HTMLTextAreaElement | null>(null);
	let hlEl = $state<HTMLDivElement | null>(null);

	const DEBOUNCE_MS = 400;

	function updateHighlight(source: string) {
		if (mode === 'template') {
			hlHtml = highlightTemplate(source);
		} else {
			hlHtml = highlightScript(source);
		}
	}

	// Reactive highlight update
	$effect(() => {
		updateHighlight(value);
	});

	async function check(source: string) {
		if (source === '') {
			error = '';
			return;
		}
		let types = contextTypes;
		if (discoverContext) {
			// Pass 1: analyze (lenient) — discover unresolved context keys
			const analysis = analyzeWithTypes(source, mode, contextTypes);
			if (analysis.ok) {
				const merged = { ...contextTypes };
				for (const key of analysis.context_keys) {
					if (!isUnknownType(key.type) && !(key.name in merged)) {
						merged[key.name] = key.type;
					}
				}
				types = merged;
			}
		}
		// Pass 2 (or only pass): typecheck (hard)
		const result = expectedTailType
			? typecheckWithTail(source, mode, types, expectedTailType)
			: typecheckWithTypes(source, mode, types);
		error = result.ok ? '' : result.message;
	}

	function scheduleCheck(source: string) {
		if (debounceTimer) clearTimeout(debounceTimer);
		debounceTimer = setTimeout(() => {
			debounceTimer = null;
			check(source);
		}, DEBOUNCE_MS);
	}

	function flushCheck(source: string) {
		if (debounceTimer) {
			clearTimeout(debounceTimer);
			debounceTimer = null;
		}
		check(source);
	}

	function handleInput(e: Event & { currentTarget: HTMLTextAreaElement }) {
		const v = e.currentTarget.value;
		oninput(v);
		scheduleCheck(v);
	}

	function handleBlur(e: FocusEvent & { currentTarget: HTMLTextAreaElement }) {
		flushCheck(e.currentTarget.value);
	}

	function handleScroll() {
		if (textareaEl && hlEl) {
			hlEl.scrollTop = textareaEl.scrollTop;
			hlEl.scrollLeft = textareaEl.scrollLeft;
		}
	}

	// Re-check when contextTypes or expectedTailType changes
	let checkKey = $derived(JSON.stringify([contextTypes, expectedTailType]));
	$effect(() => {
		void checkKey;
		if (value) scheduleCheck(value);
	});

	onDestroy(() => {
		if (debounceTimer) clearTimeout(debounceTimer);
	});

	function autogrow(el: HTMLTextAreaElement, skip: boolean) {
		if (skip) return;
		function resize() {
			el.style.height = 'auto';
			el.style.height = el.scrollHeight + 'px';
		}
		resize();
		el.addEventListener('input', resize);
		return { destroy() { el.removeEventListener('input', resize); } };
	}
</script>

<div class="sf-wrap" class:sf-unlimited={unlimited}>
	<div class="sf-editor" class:sf-unlimited={unlimited} style:--sf-rows={rows}>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="sf-hl" bind:this={hlEl} aria-hidden="true">{@html hlHtml}</div>
		<textarea
			class="sf-textarea"
			class:sf-error={!!error}
			{rows}
			{placeholder}
			{value}
			oninput={handleInput}
			onblur={handleBlur}
			onscroll={handleScroll}
			spellcheck="false"
			use:autogrow={unlimited}
			bind:this={textareaEl}
		></textarea>
	</div>
	{#if error}
		<p class="sf-error-msg">{error}</p>
	{/if}
</div>

<style>
	.sf-wrap {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.sf-wrap.sf-unlimited {
		flex: 1;
		min-height: 0;
	}
	.sf-editor {
		position: relative;
		min-height: calc(var(--sf-rows, 1) * 1.625em + 0.75rem);
	}
	.sf-editor.sf-unlimited {
		flex: 1;
		min-height: 0;
	}
	.sf-hl,
	.sf-textarea {
		box-sizing: border-box;
		font-family: var(--font-mono, ui-monospace, monospace);
		font-size: 0.75rem;
		line-height: 1.625;
		letter-spacing: normal;
		white-space: pre-wrap;
		word-break: break-all;
		tab-size: 2;
		-moz-tab-size: 2;
		padding: 0.375rem 0.5rem;
		margin: 0;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
	}
	.sf-hl {
		position: absolute;
		inset: 0;
		overflow: hidden;
		pointer-events: none;
		z-index: 0;
		color: var(--color-foreground);
		background: transparent;
		scrollbar-width: none;
	}
	.sf-unlimited .sf-hl {
		overflow: auto;
	}
	.sf-hl::-webkit-scrollbar { display: none; }
	.sf-textarea {
		position: relative;
		z-index: 1;
		width: 100%;
		resize: none;
		overflow-y: auto;
		max-height: 24rem;
		background: transparent;
		color: transparent;
		caret-color: var(--color-foreground);
		outline: none;
		transition: border-color 0.15s;
		/* Hide scrollbar so content width matches overlay exactly */
		scrollbar-width: none;
		/* Remove mobile browser default textarea styling */
		-webkit-appearance: none;
		appearance: none;
		-webkit-text-size-adjust: 100%;
	}
	.sf-textarea::-webkit-scrollbar { display: none; }
	.sf-unlimited .sf-textarea {
		position: absolute;
		inset: 0;
		height: 100%;
		max-height: none;
	}
	.sf-textarea::selection {
		background: color-mix(in oklch, var(--color-primary) 30%, transparent);
		color: transparent;
	}
	.sf-textarea:focus {
		box-shadow: 0 0 0 1px var(--color-ring);
	}
	.sf-textarea.sf-error {
		border-color: var(--color-destructive);
	}
	.sf-textarea.sf-error:focus {
		box-shadow: 0 0 0 1px var(--color-destructive);
	}
	.sf-textarea::placeholder {
		color: var(--color-muted-foreground);
	}
	.sf-error-msg {
		font-size: 0.6875rem;
		color: var(--color-destructive);
		margin: 0;
	}

	/* Syntax highlight tokens */
	.sf-hl :global(.hl-delim) { color: hsl(207 90% 60%); font-weight: 600; }
	.sf-hl :global(.hl-kw) { color: hsl(300 50% 65%); }
	.sf-hl :global(.hl-ctx) { color: hsl(158 60% 55%); }
	.sf-hl :global(.hl-str) { color: hsl(20 70% 60%); }
	.sf-hl :global(.hl-num) { color: hsl(96 40% 65%); }
	.sf-hl :global(.hl-fn) { color: hsl(50 70% 65%); }
	.sf-hl :global(.hl-pipe) { color: hsl(50 70% 65%); font-weight: 600; }
	.sf-hl :global(.hl-op) { color: var(--color-foreground); }
	.sf-hl :global(.hl-raw) { color: var(--color-muted-foreground); }

	/* Markdown tokens */
	.sf-hl :global(.hl-md-h) { color: hsl(174 60% 55%); font-weight: 600; }
	.sf-hl :global(.hl-md-b) { font-weight: 600; }
	.sf-hl :global(.hl-md-i) { font-style: italic; }
	.sf-hl :global(.hl-md-code) { color: hsl(0 60% 60%); }
	.sf-hl :global(.hl-md-link) { color: hsl(207 90% 60%); }
	.sf-hl :global(.hl-md-li) { color: hsl(174 60% 55%); }
	.sf-hl :global(.hl-md-bq) { color: var(--color-muted-foreground); font-style: italic; }
	.sf-hl :global(.hl-md-hr) { color: var(--color-muted-foreground); }
</style>
