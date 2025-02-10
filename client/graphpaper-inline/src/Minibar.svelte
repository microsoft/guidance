<!-- Mini bar graph for tracking distributions or values over time scaled from 0 to 1. -->

<script lang="ts">
  import { scaleLinear } from 'd3-scale';

  export let values;
  export let svgClass: string;
  export let rectClass: string;
  export let padding = {
    'left': 0,
    'right': 0,
    'top': 0,
    'bottom': 0
  };
  $: typedValues = values as Array<number>;

  let height = 0;
  let width = 0;
  let minVal = 0.05;
  $: xScale = scaleLinear()
    .domain([0, typedValues.length])
    .range([0, width - padding.right]);

  $: yScale = scaleLinear()
    .domain([0, 1])
    .range([height - padding.bottom, padding.top]);

  $: innerWidth = width - (padding.left + padding.right);
  $: barWidth = innerWidth / typedValues.length;
</script>

<div class="flex items-center" bind:clientHeight={height} bind:clientWidth={width}>
  <svg class={svgClass}>
    <g>
      {#each typedValues as val, i}
        <rect x={xScale(i)} y={yScale(Math.max(val, minVal))} width={barWidth-1}
              height={yScale(0) - yScale(Math.max(val, minVal))} class={rectClass}></rect>
      {/each}
    </g>
  </svg>
</div>