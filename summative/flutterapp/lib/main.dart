import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(MaterialApp(home: PredictionApp()));

class PredictionApp extends StatefulWidget {
  @override
  _PredictionAppState createState() => _PredictionAppState();
}

class _PredictionAppState extends State<PredictionApp> {
  // Add controllers for ALL variables
  final TextEditingController schoolingController = TextEditingController();
  final TextEditingController gdpController = TextEditingController();
  String result = "";

  Future<void> makePrediction() async {
    final url = Uri.parse(
      'https://life-expectancy-api-qs4d.onrender.com/predict',
    );
    try {
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "Year": 2024,
          "Status": 1,
          "Adult_Mortality": 263.0,
          "Alcohol": 0.01,
          "Hepatitis_B": 64.0,
          "BMI": 19.1,
          "Polio": 6.0,
          "Diphtheria": 65.0,
          "GDP": double.parse(gdpController.text),
          "Schooling": double.parse(schoolingController.text),
        }),
      );

      if (response.statusCode == 200) {
        setState(() {
          result =
              "Predicted Life Expectancy: ${jsonDecode(response.body)['predicted_life_expectancy']}";
        });
      } else {
        setState(() => result = "Error: Check your input ranges.");
      }
    } catch (e) {
      setState(() => result = "Error: Could not connect to API.");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Life Expectancy Predictor")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: schoolingController,
              decoration: InputDecoration(labelText: "Schooling (0-25)"),
            ),
            TextField(
              controller: gdpController,
              decoration: InputDecoration(labelText: "GDP"),
            ),
            SizedBox(height: 20),
            ElevatedButton(onPressed: makePrediction, child: Text("Predict")),
            SizedBox(height: 20),
            Text(
              result,
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}
