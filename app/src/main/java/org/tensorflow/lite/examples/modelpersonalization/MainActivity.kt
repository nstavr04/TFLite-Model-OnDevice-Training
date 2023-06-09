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
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import androidx.activity.viewModels
import org.tensorflow.lite.examples.modelpersonalization.databinding.ActivityMainBinding
import org.tensorflow.lite.examples.modelpersonalization.fragments.HelperDialog
import org.tensorflow.lite.examples.modelpersonalization.fragments.SettingFragment
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var activityMainBinding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        activityMainBinding.imgSetting.setOnClickListener {
            // Making sure that you can increase threads in settings only when in training mode
            // and not inference mode
            if (viewModel.getCaptureMode() == true) {
                SettingFragment().show(
                    supportFragmentManager,
                    SettingFragment.TAG
                )
            } else {
                Toast.makeText(
                    this, "Change the setting only available in " +
                            "training mode", Toast.LENGTH_LONG
                ).show()
            }
        }
        // Show HelperDialog when the user clicks on the helper icon
        activityMainBinding.tvHelper.setOnClickListener {
            HelperDialog().show(supportFragmentManager, HelperDialog.TAG)
        }
        // Call refresh function when user clicks on the refresh icon
        // Can implement it in future
//        activityMainBinding.refreshButton.setOnClickListener {
//            restartAppAndModel()
//        }
    }

    // Will can implement this as a reset button in future
    private fun restartAppAndModel() {
    }

    // Might be fixed but unsure
    override fun onBackPressed() {
        if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
            // Workaround for Android Q memory leak issue in IRequestFinishCallback$Stub.
            // (https://issuetracker.google.com/issues/139738913)
            finishAfterTransition()
        } else {
            super.onBackPressed()
        }
    }

    companion object {
         fun getCheckpointPath(context: Context): String {
            val checkpointDir = context.getDir("checkpoints", Context.MODE_PRIVATE)
            if (!checkpointDir.exists()) {
                checkpointDir.mkdirs()
            }
            return File(checkpointDir, "checkpoint").absolutePath
        }
    }
}
