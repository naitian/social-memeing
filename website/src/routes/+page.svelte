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
    Social Meme-ing: Measuring Linguistic Variation in Memes
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
    <a href="https://aclanthology.org/2024.naacl-long.166.pdf">📝 Read the Paper</a>
    <a href="https://github.com/naitian/social-memeing">💻 View the Code</a>
    <a href="#data">📊 Get the data</a>
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

  <Wide>
    <div style:display="flex">

      <div style:width="60%">

        <p>
          <b>We learn the semantics of meme templates without supervision.</b> We
          take advantage of the multimodal structure to learn how fill text
          aligns with the template by fine-tuning a RoBERTa model, giving
          semantic embeddings for templates which we can then cluster. 
        </p>

        <p> 
          <b>We create the <SmallCaps>SemanticMemes</SmallCaps> dataset.</b> We use
          this method to construct and make available a dataset of 3.8M Reddit memes
          grouped into semantically coherent clusters.
        </p>

        <p>
          <b>We find memes with socially meaningful variation.</b>
          Not only do subreddits prefer certain variants of a template over
          others, but they choose templates that index into a localized
          cultural knowledge, making cultural allusions to characters or
          celebrities. Click below to see examples.

        </p>

      </div>

      <div style:float="right" style:display="inline" style:width="40%">
        <Figure caption="Memes are multimodal constructions where the base image <span style='font-weight: bold; color: #E69138'>template</span> and additional text <span style='font-weight: bold; color: #3C78D8'>fills</span> both have semantic value."> <Dissection/> </Figure>
      </div>
    </div>
  </Wide>
  <Wide>
    <Figure 
      caption="Different subreddits systematically prefer some meme
      templates over other semantically equivalent ones. <b>Click on a cluster</b> to see the different variants. All examples here are statistically significantly overrepresented in their respective subreddits, p < 0.05.">
      <Variation/>
    </Figure>
  </Wide>
  <p> 
    What's more, we find that patterns of linguistic innovation  and
    acculturation that have been previously observed in text also occur with
    memes! Read the paper for more details about these experiments, as well as
    our experiments with using CLIP to train a multimodal model and more
    information about the evaluation process.
  </p>

  <h2 id="data">Explore the data</h2>
  <p>You can download the dataset with Reddit post ID, image URL, RoBERTa semantic cluster label, and template visual cluster label <a href="https://people.ischool.berkeley.edu/~naitian/social-memeing/semantic_memes_roberta.tar.gz">at this link</a> (63MB gzipped).</p>

  <p>Use the dropdown below to view some more examples of semantic clusters.</p>

  <Figure caption="Visually diverse clusters emerge even for complex semantic functions."> 
    <SemClusters/>
  </Figure>
  <h2>Conclusion</h2>
  <p>
    We hope that this work will encourage more research into the social
    language of memes, and that the <SmallCaps>SemanticMemes</SmallCaps>
    dataset will be a useful resource for future work.
  </p>

  <p> 
    To cite this work:

  </p>
  <pre><span></span>
{`@inproceedings{zhou-etal-2024-social,
  title = "Social Meme-ing: Measuring Linguistic Variation in Memes",
  author = "Zhou, Naitian  and
    Jurgens, David  and
    Bamman, David",
  editor = "Duh, Kevin  and
    Gomez, Helena  and
    Bethard, Steven",
  booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
  month = jun,
  year = "2024",
  address = "Mexico City, Mexico",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2024.naacl-long.166",
  pages = "3005--3024",
  abstract = "Much work in the space of NLP has used computational methods to explore sociolinguistic variation in text. In this paper, we argue that memes, as multimodal forms of language comprised of visual templates and text, also exhibit meaningful social variation. We construct a computational pipeline to cluster individual instances of memes into templates and semantic variables, taking advantage of their multimodal structure in doing so. We apply this method to a large collection of meme images from Reddit and make available the resulting SemanticMemes dataset of 3.8M images clustered by their semantic function. We use these clusters to analyze linguistic variation in memes, discovering not only that socially meaningful variation in meme usage exists between subreddits, but that patterns of meme innovation and acculturation within these communities align with previous findings on written language.",
}` }  </pre>
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

.content {
  font-family: Open Sans, sans-serif;
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
