import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {Chart, registerables} from 'chart.js';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements AfterViewInit {

  @ViewChild('lineCanvas', {read: ElementRef, static: false})
  private lineCanvas: ElementRef;

  predictions: any[] = [];
  results: any;
  lineChart: any;

  constructor(private http: HttpClient) {
    Chart.register(...registerables);
  }

  ngAfterViewInit() {
    this.lineChart = new Chart(this.lineCanvas.nativeElement, {
      type: 'line',
      data: {
        labels: [],
        datasets: []
      }
    });
  }

  public uploadFile(files: FileList, model: string, trainYear: string, testYear: string) {
    if(files && files.length > 0) {
      const file: File = files.item(0);
      const request: FormData = new FormData();
      request.append('file_upload', file, file.name);
      request.append('train_year', trainYear);
      request.append('test_year', testYear);

      this.http.post('http://localhost:5000/' + model, request).subscribe((response: {id: string}) => {
        console.log(response);

        this.predictions.push(response.id);
      });
    }
  }

  displayResults(predictionId) {
    this.http.get('http://localhost:5000/prediction?id=' + predictionId).subscribe((response: {
      status: string;
      meanAbsoluteDeviation: number;
      meanSquaredError: number;
      rootMeanSquaredError: number;
      train: string[];
      trainLabels: string[];
      test: string[];
      testLabels: string[];
      forecast: string[];
      forecastLabels: string[];
    }) => {
      if (response.status) {
        console.log(predictionId + ' ' + response.status);
        return;
      }

      this.results = {
        mad: response.meanAbsoluteDeviation,
        mse: response.meanSquaredError,
        rmse: response.rootMeanSquaredError,
        train: response.train,
        trainLabels: response.trainLabels,
        test: response.test,
        testLabels: response.testLabels,
        forecast: response.forecast,
        forecastLabels: response.forecastLabels
      };

      const train = this.results.trainLabels.map((e, i) => [e, response.train[i]]);

      const test = this.results.testLabels.map((e, i) => [e, response.test[i]]);

      const forecast = this.results.forecastLabels.map((e, i) => [e, response.forecast[i]]);

      this.lineChart.data = {
        labels: [... this.results.trainLabels, this.results.forecastLabels],
        datasets: [
          {
            label: 'Train',
            fill: false,
            backgroundColor: 'rgba(0,26,255,0.4)',
            borderColor: 'rgb(0,41,255)',
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'miter',
            pointBorderColor: 'rgb(0,72,255)',
            pointBackgroundColor: '#fff',
            pointBorderWidth: 1,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: 'rgb(0,40,255)',
            pointHoverBorderColor: 'rgba(220,220,220,1)',
            pointHoverBorderWidth: 2,
            pointRadius: 1,
            pointHitRadius: 10,
            data: train,
            spanGaps: false,
          },
          {
            label: 'Forecast',
            fill: false,
            backgroundColor: 'rgba(255,119,0,0.4)',
            borderColor: 'rgb(255,89,0)',
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'miter',
            pointBorderColor: 'rgb(255,101,0)',
            pointBackgroundColor: '#fff',
            pointBorderWidth: 1,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: 'rgb(255,71,0)',
            pointHoverBorderColor: 'rgba(220,220,220,1)',
            pointHoverBorderWidth: 2,
            pointRadius: 1,
            pointHitRadius: 10,
            data: forecast,
            spanGaps: false,
          },
          {
            label: 'Test',
            fill: false,
            backgroundColor: 'rgba(75,192,192,0.4)',
            borderColor: 'rgba(75,192,192,1)',
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'miter',
            pointBorderColor: 'rgba(75,192,192,1)',
            pointBackgroundColor: '#fff',
            pointBorderWidth: 1,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: 'rgba(75,192,192,1)',
            pointHoverBorderColor: 'rgba(220,220,220,1)',
            pointHoverBorderWidth: 2,
            pointRadius: 1,
            pointHitRadius: 10,
            data: test,
            spanGaps: false,
          },
        ]
      };

      this.lineChart.update();
    });
  }

}
