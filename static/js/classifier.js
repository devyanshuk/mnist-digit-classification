var canvas = $("#area")[0];
var context = canvas.getContext("2d");
context.lineWidth = 40;
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

$("#area").mousedown(function(e) {
    curX = e.clientX - area.offsetLeft;
    curY = e.clientY - area.offsetTop;
    hold = true;
        
    prevX = curX;
    prevY = curY;
    context.beginPath();
    context.moveTo(prevX, prevY);
});

$("#area").mousemove(function(e) {
    if(hold){
        curX = e.clientX - area.offsetLeft;
        curY = e.clientY - area.offsetTop;
        draw();
    }
});

$("#area").on("mouseout mouseup", function (e){
    hold = false;
});

$("#pred").click(function() {
    $.post("image/", { image: canvas.toDataURL("image/jpg", 1.0) },
    function(data) {
        var response = jQuery.parseJSON(data);
        var softmax_predictions = response.softmax;
        var prediction = response.prediction;
        var res = "<tr>";
        for (var j=0; j < 10; j++) {
            res += "<td><b>" + j + "</b></td>";
        }
        res += "</tr><tr>";
        for (var i in softmax_predictions) {
            var color = prediction == i ? 0 : 255;
            res += "<td style='background-color: rgb(" + color + ",255," + color + ")'>" + softmax_predictions[i] + "</td>";
        }
        res += "</tr>";
        $("#results").html(res);

    });
});
    
function draw(){
    context.lineTo(curX, curY);
    context.strokeStyle = beingErased ? "#ffffff" : "#000000";
    context.stroke();
}