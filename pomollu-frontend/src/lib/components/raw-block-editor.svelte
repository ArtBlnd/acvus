<script lang="ts">
	import type { RawBlock } from '$lib/types.js';
	import AcvusEngineField from './acvus-engine-field.svelte';

	let {
		block,
		onupdate,
		contextTypes = {}
	}: {
		block: RawBlock;
		onupdate: (updater: (b: RawBlock) => RawBlock) => void;
		contextTypes?: Record<string, import('$lib/type-parser.js').TypeDesc>;
	} = $props();
</script>

<div class="raw-editor">
	<AcvusEngineField
		mode={block.mode}
		value={block.text}
		oninput={(v) => onupdate((b) => ({ ...b, text: v }))}
		placeholder="Enter text..."
		unlimited
		{contextTypes}
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
