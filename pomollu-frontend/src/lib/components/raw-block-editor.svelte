<script lang="ts">
	import type { RawBlock } from '$lib/types.js';
	import type { DocumentManager } from '$lib/document-manager.svelte.js';
	import AcvusEngineField from './acvus-engine-field.svelte';

	let {
		block,
		onupdate,
		docManager,
		docKey,
	}: {
		block: RawBlock;
		onupdate: (updater: (b: RawBlock) => RawBlock) => void;
		docManager?: DocumentManager;
		docKey?: string;
	} = $props();
</script>

<div class="raw-editor">
	<AcvusEngineField
		mode={block.mode}
		value={block.text}
		oninput={(v) => onupdate((b) => ({ ...b, text: v }))}
		placeholder="Enter text..."
		unlimited
		{docManager}
		{docKey}
	/>
</div>

<style>
	.raw-editor {
		display: flex;
		flex-direction: column;
		flex: 1;
		min-height: 0;
	}
</style>
