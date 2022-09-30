import * as tf from "@tensorflow/tfjs"

export const MNIST_IMAGE_WIDTH = 28
export const MNIST_IMAGE_HEIGHT = 28
export const MNIST_IMAGE_COUNT = 65000

export interface MnistDataset {
  classes: tf.Tensor2D
  images: tf.Tensor2D
  labels: number[]
}

/**
 * Load the whole MNIST dataset into memory. Caches loaded data in local storage.
 *
 * MNIST dataset is loaded from files prepackaged for the MNIST tensorflow codelab
 * https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html
 *
 * Based on
 * https://storage.googleapis.com/tfjs-tutorials/mnist_data.js
 */
export async function loadMnistDataset(): Promise<MnistDataset> {
  const [imageData, labelData] = await Promise.all([
    getMnistImageData(),
    getMnistLabelData(),
  ])

  return tf.tidy(() => {
    const classes = tf.tensor2d(labelData, [MNIST_IMAGE_COUNT, 10])
    const labels = classes.arraySync().map(classes => classes.indexOf(1))

    const images = tf
      .tensor2d(imageData, [
        MNIST_IMAGE_COUNT,
        MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT,
      ])
      .reshape<tf.Tensor2D>([
        MNIST_IMAGE_COUNT,
        MNIST_IMAGE_WIDTH,
        MNIST_IMAGE_HEIGHT,
        1,
      ])

    return { classes, images, labels }
  })
}

export function mnistSlice(
  dataset: MnistDataset,
  start: number,
  size: number
): MnistDataset {
  return tf.tidy(() => ({
    classes: dataset.classes.slice(start, size),
    labels: dataset.labels.slice(start, start + size),
    images: dataset.images.slice(start, size),
  }))
}

const MNIST_IMAGES_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png"
const MNIST_LABELS_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8"

async function getMnistLabelData() {
  const response = await fetch(MNIST_LABELS_URL)
  return new Uint8Array(await response.arrayBuffer())
}

async function getMnistImageData() {
  const image = await getImage(MNIST_IMAGES_URL)
  return getRedImageChannel({ image, windowHeight: 5_000 })
}

async function getImage(src: string): Promise<HTMLImageElement> {
  return new Promise<HTMLImageElement>(resolve => {
    const image = new Image()
    image.crossOrigin = ""

    image.onload = () => {
      image.width = image.naturalWidth
      image.height = image.naturalHeight
      resolve(image)
    }

    image.src = src
  })
}

function getRedImageChannel({
  image,
  windowHeight,
}: {
  image: HTMLImageElement
  windowHeight: number
}) {
  const windowCanvas = document.createElement("canvas")
  const windowWidth = image.width
  windowCanvas.width = windowWidth
  windowCanvas.height = windowHeight
  const windowContext = windowCanvas.getContext("2d")

  if (!windowContext) throw new Error("Failed to create 2d canvas context")

  const data = new ArrayBuffer(image.width * image.height * 4)

  for (let y = 0; y < image.height; y += windowHeight) {
    windowContext.drawImage(
      image,
      // source:
      0,
      y,
      windowWidth,
      windowHeight,
      // destination:
      0,
      0,
      windowWidth,
      windowHeight
    )

    const windowData = new Float32Array(data, windowWidth * y * 4)
    const windowImageData = windowContext.getImageData(
      0,
      0,
      windowWidth,
      windowHeight
    )

    // copy red image channel as grayscale data
    for (let i = 0; i * 4 < windowImageData.data.length; ++i) {
      windowData[i] = windowImageData.data[i * 4] / 255
    }
  }

  return new Float32Array(data)
}
