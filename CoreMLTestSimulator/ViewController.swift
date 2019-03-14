//
//  ViewController.swift
//  CoreMLTestSimulator
//
//  Created by Ruben van den Engel on 14/03/2019.
//  Copyright © 2019 Ruben van den Engel. All rights reserved.
//

import UIKit
import Vision
import VideoToolbox
import os

typealias Prediction = (String, Double)
typealias SelectableModel = (String, Selectmodel)

class ViewController: UIViewController {

  @IBOutlet weak var imageView: UIImageView!
  @IBOutlet weak var pickerView: UIPickerView!
  @IBOutlet weak var resultText: UILabel!
  
  let pickerData: [SelectableModel] = [("MobileNet", .mobileNet), ("Resnet50", .resnet50), ("Yolov3", .yolov3), ("Inceptionv3", .inceptionv3)]
  
  override func viewDidLoad() {
    super.viewDidLoad()
    
    resultText.text = ""
    
    self.pickerView.delegate = self
    self.pickerView.dataSource = self
  }

  @IBAction func openCameraButton(_ sender: Any) {
    if UIImagePickerController.isSourceTypeAvailable(.camera) {
      let imagePicker = UIImagePickerController()
      imagePicker.delegate = self
      imagePicker.sourceType = .camera;
      imagePicker.allowsEditing = false
      self.present(imagePicker, animated: true, completion: nil)
    } else {
      let alert = UIAlertController(title: "No camera permissions", message: "You do not have the right permissions to start the camera", preferredStyle: .alert)
      alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
      
      self.present(alert, animated: true)
    }
  }
  
  func printResult(_ results: [Prediction]) {
    let resultString = results.map { (result) -> String in
      return "\(result.0): \(result.1)"
    }.joined(separator: "\n\n")
    resultText.text = resultString
  }
  
  
  func useModel(_ image: UIImage) {
    switch pickerData[pickerView.selectedRow(inComponent: 0)].1 {
    case .mobileNet:
      print("MobileNet Choosen")
      predictMobileNetUsingCoreML(image: image)
    case .resnet50:
      print("ResNet50 Choosen")
      predictResNet50UsingCoreML(image: image)
    case .yolov3:
      print("Yolov3 Choosen")
      predictYOLOUsingCoreML(image: image)
    case .inceptionv3:
      print("Inceptionv3 Choosen")
      predictInceptionv3UsingCoreML(image: image)
    }
    
//    predictInceptionv3UsingCoreML(image: image)
//    predictResNet50UsingCoreML(image: image)
//    predictYOLOUsingCoreML(image: image)
//    predictMobileNetUsingCoreML(image: image)
  }
  
  func predictInceptionv3UsingCoreML(image: UIImage) {
    let model = Inceptionv3()
    let resizeLog: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "ResizeOperations")
    let testModel: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "TestModel")
    os_signpost(.begin, log: resizeLog, name: "Resize Image")
    if let pixelBuffer = image.pixelBuffer(width: 299, height: 299) {
      os_signpost(.end, log: resizeLog, name: "Resize Image")
      os_signpost(.begin, log: testModel, name: "Predict Image")
      if let prediction = try? model.prediction(image: pixelBuffer) {
        os_signpost(.end, log: testModel, name: "Predict Image")
        let top5 = top(5, prediction.classLabelProbs)
        print(top5)
        printResult(top5)
        var imoog: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &imoog)
        imageView.image = UIImage(cgImage: imoog!)
      }
    }
  }
  
  func predictResNet50UsingCoreML(image: UIImage) {
    let model = Resnet50()
    let resizeLog: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "ResizeOperations")
    let testModel: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "TestModel")
    os_signpost(.begin, log: resizeLog, name: "Resize Image")
    if let pixelBuffer = image.pixelBuffer(width: 224, height: 224) {
      os_signpost(.end, log: resizeLog, name: "Resize Image")
      os_signpost(.begin, log: testModel, name: "Predict Image")
      if let prediction = try? model.prediction(image: pixelBuffer) {
        os_signpost(.end, log: testModel, name: "Predict Image")
        let top5 = top(5, prediction.classLabelProbs)
        print(top5)
        printResult(top5)
        var imoog: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &imoog)
        imageView.image = UIImage(cgImage: imoog!)
      }
    }
  }
  
  
  func predictYOLOUsingCoreML(image: UIImage) {
    let resizeLog: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "ResizeOperations")
    let testModel: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "TestModel")
    os_signpost(.begin, log: resizeLog, name: "Resize Image")
    if let pixelBuffer = image.pixelBuffer(width: 416, height: 416) {
      os_signpost(.end, log: resizeLog, name: "Resize Image")
      os_signpost(.begin, log: testModel, name: "Predict Image")
      let yolo = YOLOHelper()
      if let boundingBoxes = try? yolo.predict(image: pixelBuffer) {
        os_signpost(.end, log: testModel, name: "Predict Image")
        let results = boundingBoxes.map { (boundingbox) -> Prediction in
          return (yolo.labels[boundingbox.classIndex], Double(boundingbox.score * 100))
        }
        print(results)
        printResult(results)
        
        var imoog: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &imoog)
        imageView.image = UIImage(cgImage: imoog!)
      }
    }
  }
  
  func predictMobileNetUsingCoreML(image: UIImage) {
    let model = MobileNet()
    let resizeLog: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "ResizeOperations")
    let testModel: OSLog = OSLog(subsystem: "nl.rubenatwork.CoreMLTestSimulator", category: "TestModel")
    os_signpost(.begin, log: resizeLog, name: "Resize Image")
    if let pixelBuffer = image.pixelBuffer(width: 224, height: 224) {
      os_signpost(.end, log: resizeLog, name: "Resize Image")
      os_signpost(.begin, log: testModel, name: "Predict Image")
      if let prediction = try? model.prediction(data: pixelBuffer) {
        os_signpost(.end, log: testModel, name: "Predict Image")
        let top5 = top(5, prediction.prob)
        print(top5)
        printResult(top5)

        var imoog: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &imoog)
        imageView.image = UIImage(cgImage: imoog!)
      }
    }
  }
  
  func top(_ k: Int, _ prob: [String: Double]) -> [Prediction] {
    precondition(k <= prob.count)
    
    return Array(prob.map { x in (x.key, x.value) }
      .sorted(by: { a, b -> Bool in a.1 > b.1 })
      .prefix(through: k - 1))
  }
}

extension ViewController: UINavigationControllerDelegate {}

extension ViewController: UIImagePickerControllerDelegate {
  func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    let image = info[UIImagePickerController.InfoKey.originalImage] as! UIImage
    useModel(image)
    dismiss(animated: true, completion: nil)
  }
}

extension ViewController: UIPickerViewDelegate {}

extension ViewController: UIPickerViewDataSource {
  func numberOfComponents(in pickerView: UIPickerView) -> Int {
    return 1
  }
  
  func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
    return pickerData.count
  }
  
  func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
    return pickerData[row].0
  }
}

