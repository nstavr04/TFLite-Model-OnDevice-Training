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

import androidx.core.content.ContextCompat.startActivity
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import java.util.*


class MainViewModel : ViewModel() {
    private val _numThread = MutableLiveData<Int>()
    val numThreads get() = _numThread

    private val _trainingState =
        MutableLiveData(TrainingState.PREPARE)
    val trainingState get() = _trainingState

    private val _captureMode = MutableLiveData(true)
    val captureMode get() = _captureMode

    private val _numberOfSamples = MutableLiveData(TreeMap<String, Int>())
    val numberOfSamples get() = _numberOfSamples

    fun configModel(numThreads: Int) {
        _numThread.value = numThreads
    }

    fun getNumThreads() = numThreads.value

    fun setTrainingState(state: TrainingState) {
        _trainingState.value = state
    }

    fun getTrainingState() = trainingState.value

    fun setCaptureMode(isCapture: Boolean) {
        _captureMode.value = isCapture
    }

    fun getCaptureMode() = captureMode.value

    fun increaseNumberOfSample(className: String) {
        val map: TreeMap<String, Int> = _numberOfSamples.value!!
        val currentNumber: Int = if (map.containsKey(className)) {
            map[className]!!
        } else {
            0
        }
        map[className] = currentNumber + 1
        _numberOfSamples.postValue(map)
    }

        // Resetting both models on button click
//    var btnResetModels = activity!!.findViewById<Button>(R.id.btn_reset_models)
//    btnResetModels.setOnClickListener(
//    object : View.OnClickListener {
//        override fun onClick(v: View) {
//            val scenario = "default"
//            databaseHelper.emptyReplayBuffer(scenario ?: "default")
//            databaseHelper.emptyTrainingSamples(scenario ?: "default")
//            databaseHelper.emptyClassButtonImages()
//            clModel.model.trainingSamples.clear()
//            tlModel.model.trainingSamples.clear()
//            mViewModel.resetView()
//            try {
//                clModel.resetModelWeights(
//                    getActivity().openFileOutput(
//                        "clmodel_weights.edgeweights",
//                        Context.MODE_PRIVATE
//                    ).getChannel()
//                )
//                tlModel.resetModelWeights(
//                    getActivity().openFileOutput(
//                        "tlmodel_weights.edgeweights",
//                        Context.MODE_PRIVATE
//                    ).getChannel()
//                )
//                tlModel.model.loadParameters(
//                    getActivity().openFileInput("tlmodel_weights.edgeweights").getChannel()
//                )
//                clModel.model.loadParameters(
//                    getActivity().openFileInput("clmodel_weights.edgeweights").getChannel()
//                )
//            } catch (e: IOException) {
//                e.printStackTrace()
//            }
//            getActivity().finish()
//            getActivity().startActivity(getActivity().getIntent())
//        }
//    })


    fun getNumberOfSample() = numberOfSamples.value

    enum class TrainingState {
        PREPARE, TRAINING, PAUSE
    }
}
