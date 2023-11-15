<script>
let selectedCluster = -1;
const clusters = [
  {
    name: "Declarative",
    memes: [
      {
        subreddit: "r/Animemes",
        path: "/variation/declare/anime.png"
      },
      {
        subreddit: "r/memes",
        path: "/variation/declare/lisa.png"
      },
    ]
  },
  {
    name: "Comparison",
    memes: [
      {
        subreddit: "r/Animemes",
        path: "/variation/compare/anime.png"
      },
      {
        subreddit: "r/dndmemes",
        path: "/variation/compare/dnd.png"
      },
      {
        subreddit: "r/PrequelMemes",
        path: "/variation/compare/prequel.png"
      },
      {
        subreddit: "r/startrekmemes",
        path: "/variation/compare/startrek.png"
      },
      {
        subreddit: "r/memes",
        path: "/variation/compare/drake.png"
      },
    ]
  },
{
    name: "Scalar increase",
    memes: [
      {
        subreddit: "r/MinecraftMemes",
        path: "/variation/scalar/minecraft.png"
      },
      {
        subreddit: "r/dndmemes",
        path: "/variation/scalar/dnd.png"
      },
      {
        subreddit: "r/memes",
        path: "/variation/scalar/pooh.png"
      },
    ]
  },
]

</script>

<div class="clusters">
  {#each clusters as cluster, i}
    <div class="cluster" on:click={() => selectedCluster == i ? selectedCluster = -1 : selectedCluster = i}
    >
      <div class="memes">
        {#each cluster.memes as meme, j}
          <div class="meme"
            style:transform={
            selectedCluster == i ?
              `perspective(50px) translate3D(${j * 160}px, 0, 1px)`
              : `perspective(50px) translate3D(0, ${-(cluster.memes.length - j) * 2}px, ${-(cluster.memes.length - j)}px)`
            }
            style:z-index={selectedCluster == i ? 10 : -1}
            style:background-color={selectedCluster == i ? "transparent" : "white"}
          >
            <div class="overlay" style:opacity={(selectedCluster == i || selectedCluster == -1) ? 0 : 0.5}></div>
            <img src={meme.path}
            />
            <span class="subreddit"
              style:opacity={selectedCluster == i ? 1 : 0}
            >{meme.subreddit}</span>
          </div>
        {/each}
      </div>
      <div class="description"
            style:opacity={selectedCluster == -1 ? 1 : 0}
      >{cluster.name}</div>
    </div>
  {/each}
</div>


<style>

.caption {
  font-family: Open Sans, sans-serif;
}
img {
  width: 150px;
  border: 1px solid black;
  box-shadow: 0 0 10px #8884;
}

.overlay {
  pointer-events: none;
  position: absolute;
  width: 150px;
  height: 150px;
  background-color: white;
}

.clusters {
  display: flex;
  flex-direction: row;
  gap: 1em;
  font-family: Open Sans, sans-serif;
  overflow-x: scroll;
  padding: 2px;
  height: 190px;
}

.cluster {
  width: 150px;
}

.memes {
  position: relative;
  height: 150px;
}

.meme {
  position: absolute;
  transition: all 0.2s ease-in-out;
}

.description {
  width: 100%;
  text-align: center;
}

</style>
