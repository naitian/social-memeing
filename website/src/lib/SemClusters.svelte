<script> 
  const clusters = [
    {name: "Declaration", path: "/semclusters/declare.png"},
    {name: "Scalar increase", path: "/semclusters/scalar.png"},
    {name: "Similarity / parts of a whole", path: "/semclusters/similarity.png"},
    {name: "Surprise narrative", path: "/semclusters/surprise.png"},
  ]

  let selectedCluster = 0;
</script>

<div class="cluster-selector">
  {#each clusters as cluster, i}
    <button class:active={selectedCluster == i} on:click={() => selectedCluster = i}>
      {#if selectedCluster == i}
        <b>{cluster.name}</b>
      {:else}
      {cluster.name}
      {/if}
    </button>
  {/each}
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
  width: 80%;
}
</style>
