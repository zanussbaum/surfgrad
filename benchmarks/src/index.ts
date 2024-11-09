import { runBenchmark } from "./benchmark";

document.addEventListener("DOMContentLoaded", () => {
  const shaderSelect = document.getElementById(
    "shader-select",
  ) as HTMLSelectElement;
  const runButton = document.getElementById(
    "run-benchmark",
  ) as HTMLButtonElement;
  const resultsDiv = document.getElementById("results") as HTMLDivElement;
  const progressBar = document.createElement("progress");
  progressBar.style.display = "none";
  progressBar.max = 1;
  progressBar.value = 0;
  resultsDiv.parentNode!.insertBefore(progressBar, resultsDiv);

  const sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

  runButton.addEventListener("click", async () => {
    const shader = shaderSelect.value;

    resultsDiv.textContent = "Running benchmark...";
    progressBar.style.display = "block";
    progressBar.value = 0;

    try {
      const results = await runBenchmark(shader, sizes, (progress) => {
        progressBar.value = progress;
      });

      let htmlOutput = "<h2>Results:</h2>";
      htmlOutput +=
        "<table><tr><th>Size</th><th>Time (ms)</th><th>GFLOPS</th></tr>";

      for (const result of results) {
        htmlOutput += `
                    <tr>
                        <td>${result.size}x${result.size}</td>
                        <td>${result.averageTime.toFixed(2)}</td>
                        <td>${result.gflops.toFixed(2)}</td>
                    </tr>
                `;
      }

      htmlOutput += "</table>";
      resultsDiv.innerHTML = htmlOutput;
    } catch (error) {
      if (error instanceof Error) {
        resultsDiv.textContent = `Error: ${error.message}`;
      } else {
        resultsDiv.textContent = `An unknown error occurred`;
      }
    } finally {
      progressBar.style.display = "none";
    }
  });
});
