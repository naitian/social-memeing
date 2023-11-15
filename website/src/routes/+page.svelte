<script>
import SmallCaps from '$lib/SmallCaps.svelte';
import Wide from '$lib/Wide.svelte';
import Figure from '$lib/Figure.svelte';
import Variation from '$lib/Variation.svelte';
import Dissection from '$lib/Dissection.svelte';
import SemClusters from '$lib/SemClusters.svelte';

import { balancer } from "svelte-action-balancer";

</script>


<header>
  <Wide><h1 class="title" use:balancer={{ enabled: true, ratio: .5 }}>
    Social meme-ing: Measuring linguistic variation in memes
  </h1></Wide>

  <div class="authors">
    <span class="author">Naitian Zhou<sup>1</sup></span>,
    <span class="author">David Jurgens<sup>2</sup></span> and
    <span class="author">David Bamman<sup>1</sup></span>
  </div>
  <div class="affiliations">
    <span class="affiliation"> <sup>1</sup> University of California, Berkeley</span>
    <span class="affiliation"> <sup>2</sup> University of Michigan</span>
  </div>

  <div class="links">
    <a href="#">📝 Read the Paper</a>
    <a href="https://github.com/naitian/semantic-memes">💻 View the Code</a>
  </div>

</header>

<div class="content">

  <h2>Introduction</h2>
  <p>
    Much work in the space of NLP has used computational methods to explore
    sociolinguistic variation in text. In this paper, we argue that memes, as
    multimodal forms of language comprised of visual templates and text, also
    exhibit meaningful social variation.
  </p>
  <h2>Highlights</h2>

  <p>
  <b>We learn the semantics of meme templates.</b> We take advantage of the
    multimodal structure of memes to develop a method that learns template
    semantics without supervision (section 2).
  <Figure caption="Memes are multimodal constructions where the base image <span style='font-weight: bold; color: #E69138'>template</span> and additional text <span style='font-weight: bold; color: #3C78D8'>fills</span> both have semantic value."> <Dissection/> </Figure>
  </p>

  <p> 
  <b>We create the <SmallCaps>SemanticMemes</SmallCaps> dataset.</b> We use
    this method to construct and make available a dataset of 3.8M Reddit memes
    grouped into semantic clusters (section 3).
    <Figure caption="Visually diverse clusters emerge even for complex semantic functions like a surprise narrative."> 
      <SemClusters/>
    </Figure>
  </p>

  <p>
    <b>Memes have socially meaningful linguistic variation.</b>
    We find that, not only do subreddits prefer certain variants of a template
    over others, but they choose templates that index into a localized cultural
    knowledge, making cultural allusions to characters or celebrities (section 5). Click
    below to see examples.
  <Wide>
    <Figure 
      caption="Different subreddits systematically prefer some meme
      templates over other semantically equivalent ones. <b>Click on a cluster</b> to see the different variants. All examples here are statistically significantly overrepresented in their respective subreddits, p < 0.05.">
      <Variation/>
    </Figure>
  </Wide>

  </p>

  <p> 
    What's more, we find that the patterns of linguistic innovation (section 6) and
    acculturation (section 7) that has been observed in text also occurs with memes!
  </p>

  <h2>Conclusion</h2>
  <p>
    We hope that this work will encourage more research into the social
    language of memes, and that the <SmallCaps>SemanticMemes</SmallCaps>
    dataset will be a useful resource for future work.
  </p>

</div>


<style>

header {
  margin-bottom: 1em;
  text-align: center;
}

.title {
  font-size: 1.8em;
  line-height: 1.2em;
  text-align: center;
  margin: auto;
}

h2 {
  font-size: 1.2em;
}

.author {
  font-weight: bold;
}
.affiliations {
  font-style: italic;
}
.affiliation {
  margin-right: 0.5em;
}

.abstract {
  font-size: 0.8em;
}

.links a {
  font-size: 0.8em;
  color: white;
  margin-right: 1em;
  border: 1px solid #ccc;
  background-color: #333;
  border-radius: 5px;
  padding: 5px 10px;
  text-decoration: none;
  font-family: Open Sans, sans-serif;
}
</style>
