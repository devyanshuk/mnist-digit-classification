var canvas = $("#area")[0];
var context = canvas.getContext("2d");
context.lineWidth = 20;
context.lineCap = context.lineJoin = "round";
var hold = false;
var prevX, prevY, curX, curY;
var beingErased=false;


$("#erase").click(function(e) {
    beingErased = !beingErased;
    $(this).toggleClass("active");

})

$("#reset").click(function() {
    context.clearRect(0, 0, canvas.width, canvas.height);
});

$("#area").on("mousedown touchstart" ,function(e) {
    ev = e.touches ? e.touches[0] : e;
    curX = ev.clientX - area.offsetLeft;
    curY = ev.clientY - area.offsetTop;
    hold = true;
        
    prevX = curX;
    prevY = curY;
    context.beginPath();
    context.moveTo(prevX, prevY);
});

$("#area").on("mousemove touchmove", function(e){
    if(hold){
        ev = e.touches ? e.touches[0] : e;
        curX = ev.clientX - area.offsetLeft;
        curY = ev.clientY - area.offsetTop;
        draw();
    }
});

$("#area").on("mouseout mouseup touchend", function (e){
    hold = false;
});

$("#pred").click(function() {
    var dataurl = canvas.toDataURL();
    $.ajax({
        url: 'image/',
        type: 'POST',
        dataType: 'json',
        data: { 
            'data_url' : dataurl
        },
        success: function (response) {
            endresult = JSON.parse(JSON.stringify(response))
            var softmax_predictions = response.softmax;
            var prediction = response.prediction;
            var image_url = response.processed_image
            var res = "<tr><td><b>Processed-Image";
            for (var j=0; j < 10; j++) {
                res += "<td><b>" + j + "</b></td>";
            }
            res += "</tr><tr><td> <img id='preview' src='' height='28' width='28'>";
            for (var i in softmax_predictions) {
                var color = prediction == i ? 0 : 255;
                res += "<td style='background-color: rgb(" + color + ",255," + color + ")'>" + softmax_predictions[i] + "</td>";
            }
            res += "</tr>";
            $("#results").html(res);
            $("#preview").prop('src', 'data:image/png;base64,' + image_url)
        }
    });
});
    
function draw(){
    context.lineTo(curX, curY);
    context.strokeStyle = beingErased ? "#ffffff" : "#000000";
    context.stroke();
}