<script lang="ts">
	import type { ScriptBlock } from '$lib/types.js';
	import { Label } from '$lib/components/ui/label';
	import AcvusEngineField from '$lib/components/acvus-engine-field.svelte';

	let {
		block,
		onupdate,
		contextTypes = {}
	}: {
		block: ScriptBlock;
		onupdate: (updater: (b: ScriptBlock) => ScriptBlock) => void;
		contextTypes?: Record<string, import('$lib/type-parser.js').TypeDesc>;
	} = $props();
</script>

<div class="space-y-4">
	<div class="space-y-1">
		<Label>Script</Label>
		<AcvusEngineField
			mode="script"
			value={block.text}
			oninput={(v) => onupdate((b) => ({ ...b, text: v }))}
			placeholder="Script content..."
			rows={12}
			{contextTypes}
		/>
	</div>
</div>
