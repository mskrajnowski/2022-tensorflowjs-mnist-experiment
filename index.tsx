import { createRoot } from "react-dom/client"
import { useRef, useEffect } from "react"
import * as tf from "@tensorflow/tfjs"

import {
  loadMnistDataset,
  MnistDataset,
  mnistSlice,
  MNIST_IMAGE_HEIGHT,
  MNIST_IMAGE_WIDTH,
} from "./src/mnist"

main()

type Batch = {
  labels: tf.Tensor2D
  xs: tf.Tensor2D
}

type Sample = {
  label: number
  tensor: tf.Tensor2D
}

async function main() {
  console.log("loading MNIST dataset")
  const mnist = await loadMnistDataset()
  console.log("MNIST dataset loaded")

  const exampleSlice = mnistSlice(mnist, 1_000, 50)

  const reactRoot = createReactRoot()
  reactRoot.render(<SamplesGallery dataset={exampleSlice}></SamplesGallery>)
}

function SamplesGallery({ dataset }: { dataset: MnistDataset }) {
  const { labels, images } = dataset

  const samples = tf.tidy(() =>
    labels.map(
      (label, index): Sample => ({
        label,
        tensor: images
          .slice(index, 1)
          .reshape([MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT, 1]),
      })
    )
  )

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        maxWidth: `${MNIST_IMAGE_WIDTH * 10}px`,
      }}
    >
      {samples.map((sample, index) => (
        <SampleImage key={index} sample={sample} />
      ))}
    </div>
  )
}

function SampleImage({ sample }: { sample: Sample }) {
  const { label, tensor } = sample

  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas) throw new Error("Missing <canvas/>")

    tf.browser.toPixels(tensor, canvas)
  }, [tensor])

  return (
    <div>
      <canvas
        width={MNIST_IMAGE_WIDTH}
        height={MNIST_IMAGE_HEIGHT}
        ref={canvasRef}
      />
      <div style={{ textAlign: "center" }}>{label}</div>
    </div>
  )
}

function createReactRoot() {
  const rootElement = document.querySelector("#react-root")

  if (!rootElement) throw new Error("#react-root not found")

  return createRoot(rootElement)
}
