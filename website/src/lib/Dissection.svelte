<script>
  import {base} from "$app/paths";
  let toptext = "Keras";
  let bottomtext = "​"

	const timeout = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
	const backspace = async (length) => {
		for (let i = 0; i < length; ++i) {
			bottomtext = bottomtext.slice(0, bottomtext.length - 1);
			await timeout(60);
		}
	};

	const type = async (append) => {
		for (let i = 0, len = append.length; i < len; i++) {
			bottomtext = bottomtext + append[i];
			await timeout(70);
		}
	};

	const intro = async () => {
    while (true) {
      await type("Pytorch");
      await timeout(7000);
      await backspace(7);
      await timeout(3000);
      await type("Numpy");
      await timeout(7000);
      await backspace(5);
      await timeout(3000);
    }
		// await type(' and I feel fine...');
		/* await timeout(500); */
		/* dark = false; */
	};
  intro()
</script>

<svg viewBox="0 0 216 107" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linejoin:round;stroke-miterlimit:2">
    <path style="fill:none" d="M0 0h216v106.575H0z"/>
    <use xlink:href="#a" width="1200" height="1200" transform="translate(18.348 .772) scale(.99831)"/>
    <path style="fill:none;stroke:#e69138;stroke-width:1px" d="M18.348.772h104.823v104.823H18.348z"/>
    <path style="fill:#fff" d="M122.016 79.671h3.27v5.53h-3.27zM122.016 22.311h3.27v5.53h-3.27z"/>
    <path d="M121.006 73.648a2.416 2.416 0 0 0-2.415-2.415H77.409a2.416 2.416 0 0 0-2.414 2.415v19.319a2.416 2.416 0 0 0 2.414 2.415h41.182a2.416 2.416 0 0 0 2.415-2.415V73.648Z" style="fill:#3c78d8;fill-opacity:.31;stroke:#3c78d8;stroke-width:1px;stroke-linecap:round;stroke-miterlimit:1.5"/>
    <text x="76.997" y="85.784" style="font-family:'RobotoMono-Bold','Roboto Mono',monospace;font-size:10px">{bottomtext}</text>
    <text x="82.518" y="28.904" style="font-family:'RobotoMono-Bold','Roboto Mono',monospace;font-size:10px">{toptext}</text>
    <path d="M120.526 15.244a2.462 2.462 0 0 0-2.46-2.461H76.975a2.462 2.462 0 0 0-2.46 2.461v19.227a2.462 2.462 0 0 0 2.46 2.461h41.091a2.462 2.462 0 0 0 2.46-2.461V15.244Z" style="fill:#3c78d8;fill-opacity:.31;stroke:#3c78d8;stroke-width:1px;stroke-linecap:round;stroke-miterlimit:1.5"/>
    <path d="m126.826 22.757-6.3 2.1 6.3 2.1v-4.2Z" style="fill:#3c78d8"/>
    <path d="M125.566 24.857h2.165a5.397 5.397 0 0 1 5.4 5.4v32.372c0 1.433.569 2.806 1.581 3.819a5.405 5.405 0 0 0 3.819 1.581h4.09" style="fill:none;stroke:#3c78d8;stroke-width:1px;stroke-linecap:round;stroke-miterlimit:1.5"/>
    <path d="m126.826 80.106-6.3 2.1 6.3 2.1v-4.2Z" style="fill:#3c78d8"/>
    <path d="M125.566 82.206h2.165a5.397 5.397 0 0 0 5.4-5.4v-3.389a5.387 5.387 0 0 1 5.388-5.388h4.102" style="fill:none;stroke:#3c78d8;stroke-width:1px;stroke-linecap:round;stroke-miterlimit:1.5"/>
    <text x="143.961" y="16.72" style="font-family:'OpenSans-Bold','Open Sans',sans-serif;font-weight:700;font-size:12px;fill:#e69138">T<tspan x="150.072px 157.162px" y="16.72px 16.72px">em</tspan>plate</text>
    <text x="145.622" y="72.588" style="font-family:'OpenSans-Bold','Open Sans',sans-serif;font-weight:700;font-size:12px;fill:#3c78d8">Fill</text>
    <path d="m129.951 15.503-6.3-2.1 6.3-2.1v4.2Z" style="fill:#e69138"/>
    <path d="M142.621 13.403h-13.93" style="fill:none;stroke:#e69138;stroke-width:1px;stroke-linecap:round;stroke-miterlimit:1.5"/>
    <defs>
        <image id="a" width="105" height="105" xlink:href={`${base}/drake.jpeg`}/>
    </defs>
</svg>
<div class="translation">"I prefer <span class="fill">{bottomtext}</span> to <span class="fill">{toptext}</span>"</div>

<style>
svg {
  display: block;
  margin: 0 auto;
}

.fill {
  background-color: #C3D6F3;
  border-radius: 3px;
  border: 1px solid #3c78d8;
  padding: 1px 3px;
}
.translation {
  text-align: center;
}
</style>
