// Generated by view binder compiler. Do not edit!
package org.tensorflow.lite.examples.modelpersonalization.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.view.PreviewView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.coordinatorlayout.widget.CoordinatorLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;
import org.tensorflow.lite.examples.modelpersonalization.R;

public final class FragmentCameraBinding implements ViewBinding {
  @NonNull
  private final CoordinatorLayout rootView;

  @NonNull
  public final LinearLayout btnCollectSample;

  @NonNull
  public final RadioButton btnInferenceMode;

  @NonNull
  public final LinearLayout btnPauseTrain;

  @NonNull
  public final LinearLayout btnResumeTrain;

  @NonNull
  public final LinearLayout btnStartTrain;

  @NonNull
  public final RadioButton btnTrainingMode;

  @NonNull
  public final CoordinatorLayout cameraContainer;

  @NonNull
  public final LinearLayout llClassFour;

  @NonNull
  public final LinearLayout llClassOne;

  @NonNull
  public final LinearLayout llClassThree;

  @NonNull
  public final LinearLayout llClassTwo;

  @NonNull
  public final ConstraintLayout optionsLayout;

  @NonNull
  public final RadioGroup radioButton;

  @NonNull
  public final TextView tvInferenceTime;

  @NonNull
  public final TextView tvLossConsumerPause;

  @NonNull
  public final TextView tvLossConsumerResume;

  @NonNull
  public final TextView tvNumberClassFour;

  @NonNull
  public final TextView tvNumberClassOne;

  @NonNull
  public final TextView tvNumberClassThree;

  @NonNull
  public final TextView tvNumberClassTwo;

  @NonNull
  public final TextView tvPauseTitle;

  @NonNull
  public final PreviewView viewFinder;

  private FragmentCameraBinding(@NonNull CoordinatorLayout rootView,
      @NonNull LinearLayout btnCollectSample, @NonNull RadioButton btnInferenceMode,
      @NonNull LinearLayout btnPauseTrain, @NonNull LinearLayout btnResumeTrain,
      @NonNull LinearLayout btnStartTrain, @NonNull RadioButton btnTrainingMode,
      @NonNull CoordinatorLayout cameraContainer, @NonNull LinearLayout llClassFour,
      @NonNull LinearLayout llClassOne, @NonNull LinearLayout llClassThree,
      @NonNull LinearLayout llClassTwo, @NonNull ConstraintLayout optionsLayout,
      @NonNull RadioGroup radioButton, @NonNull TextView tvInferenceTime,
      @NonNull TextView tvLossConsumerPause, @NonNull TextView tvLossConsumerResume,
      @NonNull TextView tvNumberClassFour, @NonNull TextView tvNumberClassOne,
      @NonNull TextView tvNumberClassThree, @NonNull TextView tvNumberClassTwo,
      @NonNull TextView tvPauseTitle, @NonNull PreviewView viewFinder) {
    this.rootView = rootView;
    this.btnCollectSample = btnCollectSample;
    this.btnInferenceMode = btnInferenceMode;
    this.btnPauseTrain = btnPauseTrain;
    this.btnResumeTrain = btnResumeTrain;
    this.btnStartTrain = btnStartTrain;
    this.btnTrainingMode = btnTrainingMode;
    this.cameraContainer = cameraContainer;
    this.llClassFour = llClassFour;
    this.llClassOne = llClassOne;
    this.llClassThree = llClassThree;
    this.llClassTwo = llClassTwo;
    this.optionsLayout = optionsLayout;
    this.radioButton = radioButton;
    this.tvInferenceTime = tvInferenceTime;
    this.tvLossConsumerPause = tvLossConsumerPause;
    this.tvLossConsumerResume = tvLossConsumerResume;
    this.tvNumberClassFour = tvNumberClassFour;
    this.tvNumberClassOne = tvNumberClassOne;
    this.tvNumberClassThree = tvNumberClassThree;
    this.tvNumberClassTwo = tvNumberClassTwo;
    this.tvPauseTitle = tvPauseTitle;
    this.viewFinder = viewFinder;
  }

  @Override
  @NonNull
  public CoordinatorLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static FragmentCameraBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static FragmentCameraBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.fragment_camera, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static FragmentCameraBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.btnCollectSample;
      LinearLayout btnCollectSample = ViewBindings.findChildViewById(rootView, id);
      if (btnCollectSample == null) {
        break missingId;
      }

      id = R.id.btnInferenceMode;
      RadioButton btnInferenceMode = ViewBindings.findChildViewById(rootView, id);
      if (btnInferenceMode == null) {
        break missingId;
      }

      id = R.id.btnPauseTrain;
      LinearLayout btnPauseTrain = ViewBindings.findChildViewById(rootView, id);
      if (btnPauseTrain == null) {
        break missingId;
      }

      id = R.id.btnResumeTrain;
      LinearLayout btnResumeTrain = ViewBindings.findChildViewById(rootView, id);
      if (btnResumeTrain == null) {
        break missingId;
      }

      id = R.id.btnStartTrain;
      LinearLayout btnStartTrain = ViewBindings.findChildViewById(rootView, id);
      if (btnStartTrain == null) {
        break missingId;
      }

      id = R.id.btnTrainingMode;
      RadioButton btnTrainingMode = ViewBindings.findChildViewById(rootView, id);
      if (btnTrainingMode == null) {
        break missingId;
      }

      CoordinatorLayout cameraContainer = (CoordinatorLayout) rootView;

      id = R.id.llClassFour;
      LinearLayout llClassFour = ViewBindings.findChildViewById(rootView, id);
      if (llClassFour == null) {
        break missingId;
      }

      id = R.id.llClassOne;
      LinearLayout llClassOne = ViewBindings.findChildViewById(rootView, id);
      if (llClassOne == null) {
        break missingId;
      }

      id = R.id.llClassThree;
      LinearLayout llClassThree = ViewBindings.findChildViewById(rootView, id);
      if (llClassThree == null) {
        break missingId;
      }

      id = R.id.llClassTwo;
      LinearLayout llClassTwo = ViewBindings.findChildViewById(rootView, id);
      if (llClassTwo == null) {
        break missingId;
      }

      id = R.id.optionsLayout;
      ConstraintLayout optionsLayout = ViewBindings.findChildViewById(rootView, id);
      if (optionsLayout == null) {
        break missingId;
      }

      id = R.id.radioButton;
      RadioGroup radioButton = ViewBindings.findChildViewById(rootView, id);
      if (radioButton == null) {
        break missingId;
      }

      id = R.id.tvInferenceTime;
      TextView tvInferenceTime = ViewBindings.findChildViewById(rootView, id);
      if (tvInferenceTime == null) {
        break missingId;
      }

      id = R.id.tv_loss_consumer_pause;
      TextView tvLossConsumerPause = ViewBindings.findChildViewById(rootView, id);
      if (tvLossConsumerPause == null) {
        break missingId;
      }

      id = R.id.tvLossConsumerResume;
      TextView tvLossConsumerResume = ViewBindings.findChildViewById(rootView, id);
      if (tvLossConsumerResume == null) {
        break missingId;
      }

      id = R.id.tvNumberClassFour;
      TextView tvNumberClassFour = ViewBindings.findChildViewById(rootView, id);
      if (tvNumberClassFour == null) {
        break missingId;
      }

      id = R.id.tvNumberClassOne;
      TextView tvNumberClassOne = ViewBindings.findChildViewById(rootView, id);
      if (tvNumberClassOne == null) {
        break missingId;
      }

      id = R.id.tvNumberClassThree;
      TextView tvNumberClassThree = ViewBindings.findChildViewById(rootView, id);
      if (tvNumberClassThree == null) {
        break missingId;
      }

      id = R.id.tvNumberClassTwo;
      TextView tvNumberClassTwo = ViewBindings.findChildViewById(rootView, id);
      if (tvNumberClassTwo == null) {
        break missingId;
      }

      id = R.id.tvPauseTitle;
      TextView tvPauseTitle = ViewBindings.findChildViewById(rootView, id);
      if (tvPauseTitle == null) {
        break missingId;
      }

      id = R.id.view_finder;
      PreviewView viewFinder = ViewBindings.findChildViewById(rootView, id);
      if (viewFinder == null) {
        break missingId;
      }

      return new FragmentCameraBinding((CoordinatorLayout) rootView, btnCollectSample,
          btnInferenceMode, btnPauseTrain, btnResumeTrain, btnStartTrain, btnTrainingMode,
          cameraContainer, llClassFour, llClassOne, llClassThree, llClassTwo, optionsLayout,
          radioButton, tvInferenceTime, tvLossConsumerPause, tvLossConsumerResume,
          tvNumberClassFour, tvNumberClassOne, tvNumberClassThree, tvNumberClassTwo, tvPauseTitle,
          viewFinder);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}
