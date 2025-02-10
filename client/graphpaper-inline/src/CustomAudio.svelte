<script>
  import { onMount } from "svelte";
  export let audioData; // Base64 data (without the data URL header)

  let audio;
  let isPlaying = false;
  let progress = 0;
  let duration = 0;
  let currentTime = 0;
  let volume = 1;
  let waveformCanvas;

  function togglePlay() {
    if (audio.paused) {
      audio.play();
      isPlaying = true;
    } else {
      audio.pause();
      isPlaying = false;
    }
  }

  function updateProgress() {
    if (audio) {
      progress = (audio.currentTime / audio.duration) * 100;
      currentTime = audio.currentTime;
      duration = audio.duration || 0;
    }
  }

  function seek(event) {
    const seekBar = event.currentTarget;
    const seekPosition = (event.offsetX / seekBar.offsetWidth) * audio.duration;
    audio.currentTime = seekPosition;
  }

  function changeVolume(event) {
    volume = event.target.value;
    audio.volume = volume;
  }

  function formatTime(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec < 10 ? "0" : ""}${sec}`;
  }

  // When audio finishes, reset the play state
  function handleEnded() {
    isPlaying = false;
    progress = 0;
    currentTime = 0;
  }

  // Helper: convert base64 string to ArrayBuffer
  function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  // Decode the audio, downsample it, and draw the waveform onto the canvas.
  async function drawWaveform() {
    if (!audioData || !waveformCanvas) return;
    const audioContext = new AudioContext();
    const arrayBuffer = base64ToArrayBuffer(audioData);
    try {
      const decodedData = await audioContext.decodeAudioData(arrayBuffer);
      const rawData = decodedData.getChannelData(0); // use first channel

      // Ensure the canvas has the proper pixel dimensions.
      const canvas = waveformCanvas;
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      const width = canvas.width;
      const height = canvas.height;

      // Downsample the raw data to one value per pixel.
      const samples = width;
      const blockSize = Math.floor(rawData.length / samples);
      const waveform = new Array(samples);
      for (let i = 0; i < samples; i++) {
        let sum = 0;
        for (let j = 0; j < blockSize; j++) {
          sum += Math.abs(rawData[i * blockSize + j]);
        }
        waveform[i] = sum / blockSize;
      }

      // Normalize the waveform data so that the maximum amplitude maps to the full canvas height.
      const maxAmp = Math.max(...waveform);
      // Prevent division by zero
      const scale = maxAmp > 0 ? 1 / maxAmp : 1;

      // Draw the waveform: each pixel column gets a vertical bar.
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#2979ff"; // same blue as your seek bar
      for (let i = 0; i < samples; i++) {
        const x = i;
        // Normalize the amplitude and then scale to the canvas height.
        const normalizedAmp = waveform[i] * scale;
        const barHeight = normalizedAmp * height;
        // Center the bar vertically.
        const y = (height - barHeight) / 2;
        ctx.fillRect(x, y, 1, barHeight);
      }
    } catch (error) {
      console.error("Error decoding audio for waveform:", error);
    }
  }

  onMount(() => {
    drawWaveform();
  });
</script>

<!--
  New layout: We wrap the waveform canvas and seek bar in a flex-col so that
  the waveform sits directly above the seek bar.
-->
<div class="rounded-[10px] border border-gray-400 bg-white p-[10px] w-[400px]">
  <div class="flex items-center gap-[10px]">
    <!-- Play Button -->
    <div
      class="w-[40px] h-[40px] rounded-full bg-[#6c7a89] flex items-center justify-center cursor-pointer"
      on:click={togglePlay}
      on:keydown={togglePlay}
      role="button"
      tabindex="0"
      aria-label="Toggle playback"
    >
      {#if isPlaying}
        <svg class="fill-white w-[20px] h-[20px]" viewBox="0 0 24 24">
          <rect x="6" y="5" width="4" height="14" />
          <rect x="14" y="5" width="4" height="14" />
        </svg>
      {:else}
        <svg class="fill-white w-[20px] h-[20px]" viewBox="0 0 24 24">
          <polygon points="5,3 19,12 5,21" />
        </svg>
      {/if}
    </div>

    <!-- Waveform and Seek Bar Column -->
    <div class="flex flex-col flex-grow gap-1">
      <!-- Waveform Canvas -->
      <canvas bind:this={waveformCanvas} class="w-full h-12"></canvas>

      <!-- Seek Bar -->
      <div
        class="h-[6px] bg-[#ddd] rounded-[3px] cursor-pointer relative"
        on:click={seek}
        on:keydown={seek}
        role="slider"
        tabindex="0"
        aria-label="Seek"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={progress}
      >
        <div
          class="h-full bg-[#2979ff] rounded-[3px] absolute"
          style="width: {progress}%"
        ></div>
      </div>
    </div>

    <!-- Time Display -->
    <div class="text-sm text-[#555] min-w-[50px]">
      {formatTime(currentTime)} / {formatTime(duration)}
    </div>

    <!-- Volume Control -->
    <div class="flex items-center gap-[5px]">
      <svg class="w-[20px] h-[20px]" viewBox="0 0 24 24">
        <path
          d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.07c1.48-.74 2.5-2.26 2.5-4.04z"
        ></path>
      </svg>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        bind:value={volume}
        on:input={changeVolume}
        class="w-[60px]"
      />
    </div>
  </div>

  <!-- Audio Element (hidden) -->
  <audio
    bind:this={audio}
    on:timeupdate={updateProgress}
    on:ended={handleEnded}
    src={"data:audio/wav;base64," + audioData}
    class="hidden"
  ></audio>
</div>
