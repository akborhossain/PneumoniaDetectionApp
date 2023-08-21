package com.example.pneumoniadetectionapp;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.util.Base64;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.pneumoniadetections.R;
import com.example.pneumoniadetections.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Predictions extends AppCompatActivity {
    private ImageView img,xaiimg;
    private TextView result;
    BitmapDrawable bitmapDrawable;
    Bitmap bitmap;
    String imageString;
    int imageSize=150;
    @SuppressLint("WrongThread")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_predictions);
        img=findViewById(R.id.originalimg);
        xaiimg=findViewById(R.id.xaiimg);
        result=findViewById(R.id.results);
        Uri imageUri = getIntent().getParcelableExtra("image");
        try {
            Bitmap originalImage = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));

            // Maintain aspect ratio while resizing
            int targetSize = 150;
            int originalWidth = originalImage.getWidth();
            int originalHeight = originalImage.getHeight();
            float scaleFactor = Math.min(
                    (float) targetSize / originalWidth,
                    (float) targetSize / originalHeight
            );
            int newWidth = Math.round(originalWidth * scaleFactor);
            int newHeight = Math.round(originalHeight * scaleFactor);
            Bitmap resizedImage = Bitmap.createScaledBitmap(originalImage, newWidth, newHeight, true);

            classifyImage(resizedImage);
        } catch (IOException e) {
            // Handle exception
            result.setText(e.toString());
        }
        img.setImageURI(imageUri);

    }
    private String getImageString(Bitmap bitmap) {
        ByteArrayOutputStream baos=new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100,baos);
        byte[] imageBytes= baos.toByteArray();
        String encodedImage=android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return encodedImage;
    }

    @SuppressLint("SetTextI18n")
    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 150, 150, 3}, DataType.FLOAT32);
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            for (int pixelValue : intValues) {
                byteBuffer.putFloat(((pixelValue >> 16) & 0xFF) * (1.f / 255));
                byteBuffer.putFloat(((pixelValue >> 8) & 0xFF) * (1.f / 255));
                byteBuffer.putFloat((pixelValue & 0xFF) * (1.f / 255));
            }

            inputFeature0.loadBuffer(byteBuffer);
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float confidence = outputFeature0.getFloatValue(0);  // Assuming it's a single value for binary classification
            String[] classes = {"Negative", "Positive"};  // Replace with your binary class names
            String predictedClass;
            if (confidence > 0.399) {
                predictedClass = classes[1];  // Positive class
            } else {
                predictedClass = classes[0];  // Negative class
            }
            result.setText("This image is classified as Pneumonia "+predictedClass);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            result.setText(e.toString());
        }
    }
}