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
    int imageSize=128;
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
            // Resize the image to match the model's input size (128x128)
            Bitmap resizedImage = Bitmap.createScaledBitmap(originalImage, 128, 128, true);

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
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < imageSize; i ++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float confidence = outputFeature0.getFloatValue(0);  // Assuming it's a single value for binary classification
            String[] classes = {"Negative", "Positive"};  // Replace with your binary class names
            String predictedClass;
            if (confidence > 0.5) {
                predictedClass = classes[1];  // Positive class
            } else {
                predictedClass = classes[0];  // Negative class
            }
            result.setText("this is "+predictedClass);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            result.setText(e.toString());
        }
    }
}