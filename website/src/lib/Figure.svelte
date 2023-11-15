<script>
export let caption;
export let title;

import { balancer } from "svelte-action-balancer";
</script>

<figure>
  {#if title}
  <h3>{title}</h3>
  {/if}
  <slot></slot>
  {#if caption}
  <figcaption class="caption" use:balancer={{enabled: true, ratio: 0.7}}>{@html caption}</figcaption>
  {/if}
</figure>

<style>
  figure {
    margin: 2em 0;
    font-family: Open Sans, sans-serif;
  }
  .caption {
    font-size: 0.8em;
    font-style: italic;
    text-align: center;
    margin: auto;
    margin-top: 0.5em;
  }
</style>
