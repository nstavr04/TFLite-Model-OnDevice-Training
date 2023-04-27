/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.modelpersonalization

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

// This class is responsible for the training process and inference process.

class TransferLearningHelper(
    var numThreads: Int = 2,
    val context: Context,
    // A listener to receive updates on the models performance during training.
    val classifierListener: ClassifierListener?
) {

    // A tf.lite interpreter used for inference and training. (contains the signatures)
    private var interpreter: Interpreter? = null
    // A list of training samples objects which contain the bottleneck
    // and label data needed for model training (I THINK).
    // Changed from private to public
    val trainingSamples: MutableList<TrainingSample> = mutableListOf()
    // ExecutorService running the model training process
    private var executor: ExecutorService? = null

    //This lock guarantees that only one thread is performing training and
    //inference at any point in time.
    // Other threads could be adding samples to the trainingSamples list
    // or updating the UI and handling user input (I THINK)
    private val lock = Any()
    // Input image dimensions for the MobileNet model.(224, 224)
    private var targetWidth: Int = 0
    private var targetHeight: Int = 0
    // Handler running on the main thread (UI thread).
    private val handler = Handler(Looper.getMainLooper())

    // Our replayBuffer
    private val replayBuffer: MutableList<TrainingSample> = mutableListOf()

    // Used as a flag because pauseTraining is called both when we pause the training
    // and when the inference button is called. We only want to update the replayBuffer once
    private var replayBufferUpdated = false
    private var firstTrainingFlag = true

    init {
        if (setupModelPersonalization()) {
            targetWidth = interpreter!!.getInputTensor(0).shape()[2]
            targetHeight = interpreter!!.getInputTensor(0).shape()[1]
        } else {
            classifierListener?.onError("TFLite failed to init.")
        }
    }

    // Close the interpreter and executor.
    fun close() {
        executor?.shutdownNow()
        executor = null
        interpreter = null
    }

    fun pauseTraining() {
        // Update replayBuffer with samples from this training cycle

        // Apparently, pauseTraining is called when we pause the training but
        // also when we click the inference button. This is the fix to it
        if(!replayBufferUpdated){
            Log.d("PauseTraining", "Updating replay buffer")
            // Disabling these for now
            //updateReplayBuffer()
            //resetTrainingSamples()
            replayBufferUpdated = true
        }

        executor?.shutdownNow()
    }

    // Basically get the model.tflite and load it into the interpreter.
    private fun setupModelPersonalization(): Boolean {
        val options = Interpreter.Options()
        options.numThreads = numThreads
        return try {
            val modelFile = FileUtil.loadMappedFile(context, "modelnew.tflite")
            interpreter = Interpreter(modelFile, options)
            true
        } catch (e: IOException) {
            classifierListener?.onError(
                "Model personalization failed to " +
                        "initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            false
        }
    }

    // Process input image and add the output into list samples which are
    // ready for training.

    fun addSample(image: Bitmap, className: String, rotation: Int) {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }
            processInputImage(image, rotation)?.let { tensorImage ->
                val bottleneck = loadBottleneck(tensorImage)
                val newSample = TrainingSample(
                        bottleneck,
                        encoding(classes.getValue(className))
                    )
                trainingSamples.add(newSample)
            }
        }
    }

    // Start training process
    fun startTraining() {
// || firstTrainingFlag
        if (interpreter == null || firstTrainingFlag) {
            setupModelPersonalization()
            firstTrainingFlag = false

        }
        else
        {
            // Save weights
//            val checkpointPath = MainActivity.getCheckpointPath(context)
//            val saveInputs: MutableMap<String, Any> = HashMap()
//            saveInputs[SAVE_INPUT_KEY] = checkpointPath
//            val saveOutputs: MutableMap<String, Any> = HashMap()
//            saveOutputs[SAVE_OUTPUT_KEY] = checkpointPath
//            interpreter?.runSignature(saveInputs, saveOutputs, SAVE_KEY)

             setupModelPersonalization()

            // Load weights
//            val restoreInputs: MutableMap<String, Any> = HashMap()
//            restoreInputs[RESTORE_INPUT_KEY] = checkpointPath
//            val restoreOutputs: MutableMap<String, Any> = HashMap()
//            val restoredTensors = HashMap<String, FloatArray>()
//            restoreOutputs[RESTORE_OUTPUT_KEY] = restoredTensors
//            interpreter?.runSignature(restoreInputs, restoreOutputs, RESTORE_KEY)
        }

        // Reset the replayBufferUpdated flag
        replayBufferUpdated = false

        // Create new thread for training process.
        executor = Executors.newSingleThreadExecutor()
        val trainBatchSize = getTrainBatchSize()

        // The fix of this exception I think is not in the getTrainBatchSize() function
        // but rather adding both training samples and replay buffer samples in the if statement
        if (trainingSamples.size + replayBuffer.size < trainBatchSize) {
            throw RuntimeException(
                String.format(
                    "Too few samples to start training: need %d, got %d",
                    trainBatchSize, trainingSamples.size
                )
            )
        }

        Log.d("ReplayBuffer", "Replay buffer size: ${replayBuffer.size}")
        Log.d("ReplayBuffer", "Training samples size: ${trainingSamples.size}")

        // Combine the training samples and the replay buffer samples
        val combinedSamples = (trainingSamples + replayBuffer).toMutableList()
        // combinedSamples.shuffle()

        Log.d("ReplayBuffer","Combined samples size: ${combinedSamples.size}")

        executor?.execute {
            synchronized(lock) {
                var avgLoss: Float

                // Keep training until the helper pause or close.
                while (executor?.isShutdown == false) {
                    var totalLoss = 0f
                    var numBatchesProcessed = 0

                    // Shuffle training samples to reduce overfitting and
                    // variance.
                    combinedSamples.shuffle()

                    // Now trainingBatches will be called with both the training samples and the replay buffer samples
                    // The function implementation will change to adapt to this
                    trainingBatches(trainBatchSize, combinedSamples)
                        .forEach { combinedSamplesCurrentBatch ->
                            val trainingBatchBottlenecks =
                                MutableList(trainBatchSize) {
                                    FloatArray(
                                        BOTTLENECK_SIZE
                                    )
                                }

                            val trainingBatchLabels =
                                MutableList(trainBatchSize) {
                                    FloatArray(
                                        classes.size
                                    )
                                }

                            // Copy a training sample list into two different
                            // input training lists.
                            combinedSamplesCurrentBatch.forEachIndexed { index, trainingSample ->
                                trainingBatchBottlenecks[index] =
                                    trainingSample.bottleneck
                                trainingBatchLabels[index] =
                                    trainingSample.label
                            }

                            val loss = training(
                                trainingBatchBottlenecks,
                                trainingBatchLabels
                            )
                            totalLoss += loss
                            numBatchesProcessed++
                        }

                    // Calculate the average loss after training all batches.
                    avgLoss = totalLoss / numBatchesProcessed
                    handler.post {
                        classifierListener?.onLossResults(avgLoss)
                    }

                }
            }
        }
    }

    // Used for debugging to check the weights after training
    private fun getModelWeightsHead(bottlenecks: MutableList<FloatArray>): FloatArray {
        val wsSize = BOTTLENECK_SIZE * NUM_CLASSES * 4 // 4 bytes per float
        val wsBuffer = ByteBuffer.allocateDirect(wsSize).order(ByteOrder.nativeOrder())

        val inputs: MutableMap<String, Any> = HashMap()
        // Used just because we need an input. Has no use
        inputs[TRAINING_INPUT_BOTTLENECK_KEY] = bottlenecks.toTypedArray()

        val outputs: MutableMap<String, Any> = HashMap()
        outputs["ws"] = wsBuffer

        // Get weights and biases as defined in our signature in the model.
        interpreter?.runSignature(inputs, outputs, "initialize")

        val wsArray = FloatArray(BOTTLENECK_SIZE * NUM_CLASSES)
        wsBuffer.rewind()
        wsBuffer.asFloatBuffer().get(wsArray)

        return wsArray
    }

    // Runs one training step with the given bottleneck batches and labels
    // and return the loss number.
    private fun training(
        bottlenecks: MutableList<FloatArray>,
        labels: MutableList<FloatArray>
    ): Float {

        val inputs: MutableMap<String, Any> = HashMap()
        inputs[TRAINING_INPUT_BOTTLENECK_KEY] = bottlenecks.toTypedArray()
        inputs[TRAINING_INPUT_LABELS_KEY] = labels.toTypedArray()

        val outputs: MutableMap<String, Any> = HashMap()
        val loss = FloatBuffer.allocate(1)
        outputs[TRAINING_OUTPUT_KEY] = loss

        // Training as defined in our signature in the model.
        interpreter?.runSignature(inputs, outputs, TRAINING_KEY)

        Log.d("Loss", "Loss is ${loss.get(0)}")

        // Get weights and biases after training
        val weights = getModelWeightsHead(bottlenecks)
        // Use or print the weights and biases as needed
        Log.d("WeightsAndBiases", "Weights: ${weights.contentToString()}")

        return loss.get(0)
    }

    // Invokes inference on the given image batches.
    fun classify(bitmap: Bitmap, rotation: Int) {
        processInputImage(bitmap, rotation)?.let { image ->
            synchronized(lock) {
                if (interpreter == null) {
                    setupModelPersonalization()
                }

                // Inference time is the difference between the system time at the start and finish of the
                // process
                var inferenceTime = SystemClock.uptimeMillis()

                val inputs: MutableMap<String, Any> = HashMap()
                inputs[INFERENCE_INPUT_KEY] = image.buffer

                val outputs: MutableMap<String, Any> = HashMap()
                val output = TensorBuffer.createFixedSize(
                    intArrayOf(1, 4),
                    DataType.FLOAT32
                )
                outputs[INFERENCE_OUTPUT_KEY] = output.buffer

                interpreter?.runSignature(inputs, outputs, INFERENCE_KEY)
                val tensorLabel = TensorLabel(classes.keys.toList(), output)
                val result = tensorLabel.categoryList

                inferenceTime = SystemClock.uptimeMillis() - inferenceTime

                classifierListener?.onResults(result, inferenceTime)
            }
        }
    }

    // Loads the bottleneck feature from the given image array.
    private fun loadBottleneck(image: TensorImage): FloatArray {
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[LOAD_BOTTLENECK_INPUT_KEY] = image.buffer
        val outputs: MutableMap<String, Any> = HashMap()
        val bottleneck = Array(1) { FloatArray(BOTTLENECK_SIZE) }
        outputs[LOAD_BOTTLENECK_OUTPUT_KEY] = bottleneck
        interpreter?.runSignature(inputs, outputs, LOAD_BOTTLENECK_KEY)
        return bottleneck[0]
    }

    // Preprocess the image and convert it into a TensorImage for classification.
    private fun processInputImage(
        image: Bitmap,
        imageRotation: Int
    ): TensorImage? {
        val height = image.height
        val width = image.width
        val cropSize = min(height, width)
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    targetHeight,
                    targetWidth,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(NormalizeOp(0f, 255f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }

    // encode the classes name to float array
    private fun encoding(id: Int): FloatArray {
        val classEncoded = FloatArray(4) { 0f }
        classEncoded[id] = 1f
        return classEncoded
    }

    // Training model expected batch size.
    // We can ask as many samples as we want
    private fun getTrainBatchSize(): Int {

        Log.d("TrainBatch", "Training samples: ${trainingSamples.size}")
        Log.d("TrainBatch", "Replay buffer: ${replayBuffer.size}")

        // Added replayBuffer sizes here too
        return min(
            max( /* at least one sample needed */1, trainingSamples.size + replayBuffer.size),
            EXPECTED_BATCH_SIZE
        )
    }

    // Constructs an iterator that iterates over training sample batches.
    // Altered the function to include replayBuffer samples as well
    private fun trainingBatches(trainBatchSize: Int, samples: List<TrainingSample>): Iterator<List<TrainingSample>> {

        return object : Iterator<List<TrainingSample>> {
            private var nextIndex = 0

            override fun hasNext(): Boolean {
                return nextIndex < samples.size
            }

            override fun next(): List<TrainingSample> {
                val fromIndex = nextIndex
                val toIndex: Int = nextIndex + trainBatchSize
                nextIndex = toIndex
                return if (toIndex >= samples.size) {
                    // To keep batch size consistent, last batch may include some elements from the
                    // next-to-last batch.
                    samples.subList(
                        samples.size - trainBatchSize,
                        samples.size
                    )
                } else {
                    samples.subList(fromIndex, toIndex)
                }
            }
        }
    }

    private fun updateReplayBuffer(){
        // Portion of trainingSamples to add to replayBuffer
        // I could zero this if I want to disable replayBuffer
        val portion = 0.25

        val samplesToAdd = (trainingSamples.size * portion).toInt()

        // Might not be necessary
        trainingSamples.shuffle()

        Log.d("ReplayBuffer","Number of trainingSamples before updating replayBuffer are: ${trainingSamples.size}")

        val samplesToAddToReplayBuffer = trainingSamples.subList(0, samplesToAdd)

        // If samplesToAddToReplayBuffer are more than REPLAY_BUFFER_SIZE we will remove some
        if (samplesToAddToReplayBuffer.size > REPLAY_BUFFER_SIZE){
            samplesToAddToReplayBuffer.subList(0, samplesToAddToReplayBuffer.size - REPLAY_BUFFER_SIZE)
        }

        // Add them to replayBuffer
        replayBuffer.addAll(samplesToAddToReplayBuffer)

        Log.d("ReplayBuffer", "Adding ${samplesToAddToReplayBuffer.size} samples to replay buffer")
        Log.d("ReplayBuffer", "Replay buffer size before removing extra samples is now: ${replayBuffer.size}")

        // We randomly remove samples from the replay buffer
        // CAUTION HERE
        // We basically add the new samples to the replay buffer
        // and then remove from the buffer until it reaches it's
        // maximum size. This means that probably some new samples
        // will be removed instantly.
        if(replayBuffer.size > REPLAY_BUFFER_SIZE){

            val samplesToRemove = replayBuffer.size - REPLAY_BUFFER_SIZE

            Log.d("ReplayBuffer", "We will remove $samplesToRemove samples from replay buffer")

            // We remove the excess samples
            for(i in 0 until samplesToRemove){
                val randomIndex = Random.nextInt(replayBuffer.size)
                val removedSample = replayBuffer.removeAt(randomIndex)
                Log.d("ReplayBuffer", "Removing samples at index $randomIndex from replay buffer")
            }
        }

        Log.d("ReplayBuffer", "Replay buffer size after removing extra samples is now: ${replayBuffer.size}")
    }

    // Still unsure about this but probably we want to remove the samples from the list after
    // training because the replayBuffer will retain the previous knowledge
    public fun resetTrainingSamples(){
        trainingSamples.clear()
    }

    public fun clearReplayBuffer(){
        replayBuffer.clear()
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(results: List<Category>?, inferenceTime: Long)
        fun onLossResults(lossNumber: Float)
    }

    companion object {
        const val CLASS_ONE = "1"
        const val CLASS_TWO = "2"
        const val CLASS_THREE = "3"
        const val CLASS_FOUR = "4"
        private val classes = mapOf(
            CLASS_ONE to 0,
            CLASS_TWO to 1,
            CLASS_THREE to 2,
            CLASS_FOUR to 3
        )
        private const val LOAD_BOTTLENECK_INPUT_KEY = "feature"
        private const val LOAD_BOTTLENECK_OUTPUT_KEY = "bottleneck"
        private const val LOAD_BOTTLENECK_KEY = "load"

        private const val TRAINING_INPUT_BOTTLENECK_KEY = "bottleneck"
        private const val TRAINING_INPUT_LABELS_KEY = "label"
        private const val TRAINING_OUTPUT_KEY = "loss"
        private const val TRAINING_KEY = "train"

        private const val INFERENCE_INPUT_KEY = "feature"
        private const val INFERENCE_OUTPUT_KEY = "output"
        private const val INFERENCE_KEY = "infer"

        private const val SAVE_INPUT_KEY = "checkpoint_path"
        private const val SAVE_OUTPUT_KEY = "checkpoint_path"
        private const val SAVE_KEY = "save"

        private const val RESTORE_INPUT_KEY = "checkpoint_path"
        private const val RESTORE_OUTPUT_KEY = "restored_tensors"
        private const val RESTORE_KEY = "restore"

        private const val NUM_CLASSES = 4
        private const val BOTTLENECK_SIZE = 1 * 7 * 7 * 1280
        private const val EXPECTED_BATCH_SIZE = 20
        private const val TAG = "ModelPersonalizationHelper"

        // Replay buffer size
        private const val REPLAY_BUFFER_SIZE = 100
    }

    data class TrainingSample(val bottleneck: FloatArray, val label: FloatArray)
}
