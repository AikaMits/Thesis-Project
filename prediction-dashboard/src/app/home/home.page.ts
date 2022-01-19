import {AfterViewInit, Component, ElementRef, ViewChild} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Chart, registerables} from "chart.js";

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements AfterViewInit {

  predictions: any[] = [];

  results: any;

  @ViewChild('lineCanvas', {read: ElementRef, static: false})
  private lineCanvas: ElementRef;

  lineChart: any;

  constructor(private http: HttpClient) {
    Chart.register(...registerables);
  }

  ngAfterViewInit() {
    this.lineChart = new Chart(this.lineCanvas.nativeElement, {
      type: "line",
      data: {
        labels: [],
        datasets: []
      }
    });
  }

  public uploadFile(files: FileList, model: string, train_year: string, test_year: string) {
    if(files && files.length > 0) {
      let file : File = files.item(0);
      let request: FormData = new FormData();
      request.append('file_upload', file, file.name);
      request.append('train_year', train_year);
      request.append('test_year', test_year);

      this.http.post('http://localhost:5000/' + model, request).subscribe(response => {
        console.log(response)

        this.predictions.push(response['id'])
      });
    }
  }

  displayResults(prediction_id) {
    this.http.get('http://localhost:5000/prediction?id=' + prediction_id).subscribe(response => {
      if (response['status']) {
        console.log(prediction_id + ' ' + response['status'])
        return;
      }

      this.results = {
        mpa: response['mpa'],
        mse: response['mse'],
        mape: response['mape'],
        train: response['train'],
        train_labels: response['train_labels'],
        test: response['test'],
        test_labels: response['test_labels'],
        forecast: response['forecast'],
        forecast_labels: response['forecast_labels']
      }

      const train = this.results.train_labels.map(function(e, i) {
        return [e, response['train'][i]];
      });

      const test = this.results.test_labels.map(function(e, i) {
        return [e, response['test'][i]];
      });

      const forecast = this.results.forecast_labels.map(function(e, i) {
        return [e, response['forecast'][i]];
      });

      this.lineChart.data = {
        labels: [... this.results.train_labels, this.results.forecast_labels],
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
      }

      this.lineChart.update();
    });
  }

}
