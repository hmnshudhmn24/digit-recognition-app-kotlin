
package com.example.digitrecognition

import android.graphics.*
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {
    private lateinit var tflite: Interpreter
    private var drawPath = Path()
    private var bitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888)
    private var canvas = Canvas(bitmap)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tflite = Interpreter(loadModelFile())
        setContent { DigitRecognitionApp() }
    }

    @Composable
    fun DigitRecognitionApp() {
        var result by remember { mutableStateOf("Draw a digit") }
        Column(
            Modifier
                .fillMaxSize()
                .background(Color.White)
                .padding(16.dp)
        ) {
            Text(result, style = MaterialTheme.typography.headlineMedium)
            Spacer(modifier = Modifier.height(16.dp))
            Box(
                modifier = Modifier
                    .size(280.dp)
                    .background(Color.LightGray)
                    .pointerInput(Unit) {
                        detectDragGestures(
                            onDragStart = { offset ->
                                drawPath.moveTo(offset.x, offset.y)
                            },
                            onDrag = { change, _ ->
                                drawPath.lineTo(change.position.x, change.position.y)
                                invalidateCanvas()
                            }
                        )
                    }
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawContext.canvas.nativeCanvas.drawPath(drawPath, Paint().apply {
                        color = android.graphics.Color.BLACK
                        strokeWidth = 20f
                        style = Paint.Style.STROKE
                    })
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
            Button(onClick = { result = predictDigit() }) {
                Text("Predict")
            }
            Button(onClick = {
                clearCanvas()
                result = "Draw a digit"
            }) {
                Text("Clear")
            }
        }
    }

    private fun invalidateCanvas() {
        canvas.drawColor(android.graphics.Color.WHITE)
        canvas.drawPath(drawPath, Paint().apply {
            color = android.graphics.Color.BLACK
            strokeWidth = 20f
            style = Paint.Style.STROKE
        })
    }

    private fun predictDigit(): String {
        val scaled = Bitmap.createScaledBitmap(bitmap, 28, 28, true)
        val byteBuffer = ByteBuffer.allocateDirect(28 * 28 * 4).order(ByteOrder.nativeOrder())
        for (y in 0 until 28) {
            for (x in 0 until 28) {
                val pixel = scaled.getPixel(x, y)
                val value = (255 - Color.red(pixel)).toFloat() / 255.0f
                byteBuffer.putFloat(value)
            }
        }
        val output = Array(1) { FloatArray(10) }
        tflite.run(byteBuffer, output)
        val maxIdx = output[0].indices.maxByOrNull { output[0][it] } ?: -1
        return "Prediction: $maxIdx"
    }

    private fun clearCanvas() {
        drawPath.reset()
        bitmap.eraseColor(android.graphics.Color.WHITE)
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = assets.openFd("mnist.tflite")
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
