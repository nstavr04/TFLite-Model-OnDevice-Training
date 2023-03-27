// Generated by view binder compiler. Do not edit!
package org.tensorflow.lite.examples.modelpersonalization.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatImageButton;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;
import org.tensorflow.lite.examples.modelpersonalization.R;

public final class FragmentSettingBinding implements ViewBinding {
  @NonNull
  private final RelativeLayout rootView;

  @NonNull
  public final Button btnCancel;

  @NonNull
  public final Button btnConfirm;

  @NonNull
  public final RelativeLayout rlThreads;

  @NonNull
  public final AppCompatImageButton threadsMinus;

  @NonNull
  public final AppCompatImageButton threadsPlus;

  @NonNull
  public final TextView threadsValue;

  private FragmentSettingBinding(@NonNull RelativeLayout rootView, @NonNull Button btnCancel,
      @NonNull Button btnConfirm, @NonNull RelativeLayout rlThreads,
      @NonNull AppCompatImageButton threadsMinus, @NonNull AppCompatImageButton threadsPlus,
      @NonNull TextView threadsValue) {
    this.rootView = rootView;
    this.btnCancel = btnCancel;
    this.btnConfirm = btnConfirm;
    this.rlThreads = rlThreads;
    this.threadsMinus = threadsMinus;
    this.threadsPlus = threadsPlus;
    this.threadsValue = threadsValue;
  }

  @Override
  @NonNull
  public RelativeLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static FragmentSettingBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static FragmentSettingBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.fragment_setting, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static FragmentSettingBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.btnCancel;
      Button btnCancel = ViewBindings.findChildViewById(rootView, id);
      if (btnCancel == null) {
        break missingId;
      }

      id = R.id.btnConfirm;
      Button btnConfirm = ViewBindings.findChildViewById(rootView, id);
      if (btnConfirm == null) {
        break missingId;
      }

      id = R.id.rlThreads;
      RelativeLayout rlThreads = ViewBindings.findChildViewById(rootView, id);
      if (rlThreads == null) {
        break missingId;
      }

      id = R.id.threads_minus;
      AppCompatImageButton threadsMinus = ViewBindings.findChildViewById(rootView, id);
      if (threadsMinus == null) {
        break missingId;
      }

      id = R.id.threads_plus;
      AppCompatImageButton threadsPlus = ViewBindings.findChildViewById(rootView, id);
      if (threadsPlus == null) {
        break missingId;
      }

      id = R.id.threads_value;
      TextView threadsValue = ViewBindings.findChildViewById(rootView, id);
      if (threadsValue == null) {
        break missingId;
      }

      return new FragmentSettingBinding((RelativeLayout) rootView, btnCancel, btnConfirm, rlThreads,
          threadsMinus, threadsPlus, threadsValue);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}
