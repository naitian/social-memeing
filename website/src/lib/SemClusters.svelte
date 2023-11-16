<script> 
  import {base} from "$app/paths";
  const clusters = [
    {name: "Surprise narrative", path: `${base}/semclusters/surprise.png`},
    {name: "Declaration", path: `${base}/semclusters/declare.png`},
    {name: "Scalar increase", path: `${base}/semclusters/scalar.png`},
    {name: "Similarity / parts of a whole", path: `${base}/semclusters/similarity.png`},
    {name: "Concern", path: `${base}/semclusters/concern.png`},
    {name: "Me and the boys", path: `${base}/semclusters/me_and_the_boys.png`},
    {name: "Unpopular opinion", path: `${base}/semclusters/unpopular.png`},
    {name: "Outsider", path: `${base}/semclusters/outsider.png`},
  ]

  let selectedCluster = 0;
</script>

<div class="cluster-selector">
  Semantic cluster: <select bind:value={selectedCluster}>
    {#each clusters as cluster, i}
      <option value={i}>{cluster.name}</option>
    {/each}
  <!-- {#each clusters as cluster, i} -->
  <!--   <button class:active={selectedCluster == i} on:click={() => selectedCluster = i}> -->
  <!--     {#if selectedCluster == i} -->
  <!--       <b>{cluster.name}</b> -->
  <!--     {:else} -->
  <!--     {cluster.name} -->
  <!--     {/if} -->
  <!--   </button> -->
  <!-- {/each} -->
</div>
<div>
  {#each clusters as cluster, i}
    {#if  selectedCluster == i}
      <img src={cluster.path} />
    {/if}
  {/each}
</div>


<style>

div {
  position: relative;
  text-align: center;
}
img {
  width: 100%;
}
</style>
